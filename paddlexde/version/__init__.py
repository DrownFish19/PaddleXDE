import os

from paddlexde.version import git

commit = "unknown"

paddlexde_dir = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
if commit.endswith("unknown") and git.is_git_repo(paddlexde_dir) and git.have_git():
    commit = git.git_revision(paddlexde_dir).decode("utf-8")
    if git.is_dirty(paddlexde_dir):
        commit += ".dirty"
del paddlexde_dir


__all__ = ["show"]


def show():
    """Get the corresponding commit id of paddlexde.

    Returns:
        The commit-id of paddlexde will be output.

        full_version: version of paddlexde


    Examples:
        .. code-block:: python

            import paddlexde

            paddlexde.version.show()
            # commit: 1ef5b94a18773bb0b1bba1651526e5f5fc5b16fa

    """
    print("commit:", commit)
