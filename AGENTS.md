# NuSeg Execution Rules

## Environment roles

- This repository on macOS is the primary Codex code-editing workspace.
- Do not run formal GPU training on the Mac.
- The GPU execution host is available through SSH host alias `nuseg-server`.
- The remote repository path is `/hy-tmp/NuSeg`.

## Remote execution

For GPU or server-environment tasks, execute commands through:

ssh nuseg-server 'cd /hy-tmp/NuSeg && <command>'

Before executing code remotely, verify that the remote repository contains
the intended commit/version.

## Repository synchronization

Source code is synchronized through GitHub.

Do not transfer or commit:

- datasets
- model weights
- checkpoints
- Hugging Face caches
- Conda environments
- large generated artifacts

These remain on the GPU server.

## GPU usage

Before formal GPU experiments:

1. Check `nvidia-smi`.
2. Confirm the intended GPU count.
3. Confirm the exact experiment command and output directory.
4. Run CPU/lightweight audit first when applicable.
5. Do not overwrite existing checkpoints or experiment results.

## Cleanup

Temporary reproducible files may be cleaned only after successful validation.

Do not automatically delete:

- checkpoints
- final logs
- result JSON files
- experiment configs
- source code
- datasets
- failed-run evidence
