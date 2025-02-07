from pdm.backend.hooks.version import SCMVersion

def format_version(version) -> str:
    """
    Format the version number of the package. This function is only supposed to be
    called by the pdm-backend to get the version number of the package.
    """
    if version.distance is None:
        return str(version.version)
    else:
        return f"{version.version}.post{version.distance}"