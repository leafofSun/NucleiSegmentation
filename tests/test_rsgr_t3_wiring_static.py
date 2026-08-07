from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_SOURCE = ROOT / "test.py"
SAM_SOURCE = ROOT / "segment_anything/modeling/sam.py"
RSGR_SOURCE = ROOT / "segment_anything/modeling/rsgr.py"
MATERIALIZER_SOURCE = ROOT / "tools/materialize_rsgr_bank.py"
VERIFIER_SOURCE = ROOT / "tools/verify_rsgr_bank_roundtrip.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


class TestRSGRT3StaticWiring(unittest.TestCase):
    def test_all_model_accepted_rsgr_cli_args_are_wired(self):
        test_tree = _parse(TEST_SOURCE)
        build = _function(test_tree, "_build_test_model")
        text_sam_calls = [
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TextSam"
        ]
        self.assertEqual(len(text_sam_calls), 1)
        wired = {keyword.arg for keyword in text_sam_calls[0].keywords}

        parser_args = {
            node.args[0].value.removeprefix("--")
            for node in ast.walk(test_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("--rsgr_")
        }
        self.assertEqual(len(parser_args), 14)

        sam_tree = _parse(SAM_SOURCE)
        text_sam = next(
            node
            for node in sam_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TextSam"
        )
        initializer = next(
            node
            for node in text_sam.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        accepted = {argument.arg for argument in initializer.args.args}
        model_rsgr_args = parser_args & accepted
        loss_only_args = parser_args - accepted
        self.assertEqual(
            loss_only_args,
            {"rsgr_local_attr_weight", "rsgr_semantic_align_weight"},
        )
        self.assertEqual(len(model_rsgr_args), 12)
        self.assertTrue(model_rsgr_args <= wired)
        self.assertIn("enable_rsgr", wired)
        self.assertTrue(loss_only_args.isdisjoint(wired))

    def test_post_checkpoint_silent_degradation_guards_are_present(self):
        source = TEST_SOURCE.read_text(encoding="utf-8")
        self.assertIn("enable_rsgr=True but model.rsgr is None", source)
        self.assertIn("checkpoint overrode CLI-selected RSGR prototype buffers", source)
        self.assertIn("[TEST_CONFIG] RSGR Runtime Configuration (post-checkpoint)", source)
        for field in (
            "rsgr_enabled=",
            "rsgr_module_built=",
            "rsgr_prototype_path=",
            "rsgr_bank_sha256=",
            "rsgr_active_prototype_sha256=",
        ):
            self.assertIn(field, source)

    def test_loader_guards_backend_schema_attribute_and_level_order(self):
        source = RSGR_SOURCE.read_text(encoding="utf-8")
        for fragment in (
            'metadata.get("backend") != "conch"',
            'metadata.get("schema_sha256") != sha256_file(schema_path)',
            'metadata.get("attribute_names") != expected_names',
            'metadata.get("class_names") != expected_classes',
            'payload.get("attribute_names") != expected_names',
            'payload.get("class_names") != expected_classes',
        ):
            self.assertIn(fragment, source)

    def test_materializer_and_verifier_keep_the_frozen_contract(self):
        materializer = MATERIALIZER_SOURCE.read_text(encoding="utf-8")
        verifier = VERIFIER_SOURCE.read_text(encoding="utf-8")
        for digest in (
            "de4413374061d3886fc87288ff48c46ea5f07d00268aaf191c7328d74f55eaa3",
            "ca28900b8650ec49974da776bdc2bef0e9408f42421e6f7aee5d4a32a34786a8",
            "cb5cfb2d79d05cbeeef28efa5a25bb1252b287ed497c231929d8447308aeea0d",
            "a10944ad06cffdf70742c93ed2c6570ec32b8810ea77ac013faacd04c0cab7f1",
            "01c8dfc779811592207df7b678b84bb192a42aebd00b18748eb09e24d0126e79",
        ):
            self.assertIn(digest, materializer)
        self.assertNotIn("import conch", materializer.lower())
        self.assertNotIn("encode_with_project_conch_path", materializer)
        self.assertNotIn("encode_text", materializer)
        self.assertIn("load_prototype_banks(", verifier)
        self.assertIn("maximum_difference != 0.0", verifier)
        self.assertIn("absolute_error <= 1e-6", verifier)


if __name__ == "__main__":
    unittest.main()
