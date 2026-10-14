"""Development-only ASGI application; never an authenticated service entrypoint.

Do not point an installed production service at this module. W0-I1a source
admission and W0-I2 first-install/continuation authorization are NOT applied.
"""

from suzka.api.server import create_app

# Intentional development convenience: reads ambient development settings only
# when the explicitly named development ASGI module is imported.
app = create_app()
