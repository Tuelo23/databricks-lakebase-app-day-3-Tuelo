"""
One-time setup: store the Alpaca paper-trading keys.

Scopes `massive` and `database` already exist from Day 1, and the
Massive API key + Lakebase URL are already stored, so those sections
are commented out. create_scope on an existing scope raises
RESOURCE_ALREADY_EXISTS and would abort before the Alpaca keys land.
"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import getpass

w = WorkspaceClient()

# --- Already done on Day 1 -------------------------------------------
# w.secrets.create_scope(scope="massive")
# w.secrets.put_secret(
#     scope="massive",
#     key="api-key",
#     string_value=getpass.getpass("Paste your Massive API key: ")
# )
# w.secrets.create_scope(scope="database")
# w.secrets.put_secret(
#     scope="database",
#     key="lakebase-url",
#     string_value=getpass.getpass("Paste your lakebase url")
# )
# ---------------------------------------------------------------------

# --- New for Day 3: Alpaca paper-trading keys ------------------------
w.secrets.put_secret(
    scope="database",
    key="alpaca-key-id",
    string_value=getpass.getpass("Paste your Alpaca Key ID: ")
)

w.secrets.put_secret(
    scope="database",
    key="alpaca-secret-key",
    string_value=getpass.getpass("Paste your Alpaca Secret Key: ")
)

# Re-applying ACLs is harmless and ensures the app can read the scopes
w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)

w.secrets.put_acl(
    scope="massive",
    principal="users",
    permission=workspace.AclPermission.READ,
)

print(sorted(s.key for s in w.secrets.list_secrets("database")))
