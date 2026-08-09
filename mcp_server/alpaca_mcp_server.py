"""
Alpaca Markets paper-trading MCP server.

Exposes paper-trading tools over MCP (Model Context Protocol) so a
Databricks Agent Bricks agent can call them like any other tool:
    - get_quote(symbol)
    - place_trade(account_id, symbol, side, quantity)
    - get_positions(account_id)
    - get_account_summary(account_id)
    - get_order_history(account_id, limit)
    - get_balance(account_id)
    - get_current_user()

These tools are backed by Alpaca Markets' real, hosted paper-trading
account (see alpaca_broker.py), so students can safely wire an Agent
Bricks agent to place real (but fake-money) trades without a real
brokerage account or risk of real money moving. account_id is accepted
for signature compatibility but is not used to select an account - Alpaca
paper trading is one account per API key pair.

Swap-in-a-real-broker note: to point this at a different broker instead,
keep the same 5 tool signatures below and replace the alpaca_broker.*
calls inside each tool with calls to that broker's SDK/API - the MCP
surface for the agent does not need to change. The original Lakebase-
simulated engine is preserved in paper_broker.py for reference.

Deploy this as its own Databricks App (same app.yaml + FastMCP entrypoint
pattern documented at
https://docs.databricks.com/aws/en/agents/mcp-tools/custom-mcp), separate
from the dashboard app, so an Agent Bricks agent (or any MCP client) can
register its URL as an external MCP server.

Run locally:
    python alpaca_mcp_server.py
"""

import os
import logging
import random
import time
import json
import uuid
from functools import wraps
from contextvars import ContextVar

from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

import alpaca_broker
import massive_broker
import lakebase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpaca-mcp-server")

# Pending trades awaiting confirmation
_PENDING_TRADES = {}

def trace_mcp_call(func):
    """Decorator to trace MCP tool calls to Lakebase."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        session_id = str(uuid.uuid4())
        tool_name = func.__name__
        start_time = time.time()
        status = 'success'
        error_message = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            status = 'error'
            error_message = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            try:
                # Capture arguments
                import inspect
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                
                lakebase.run_write(
                    """
                    INSERT INTO mcp_call_traces 
                    (session_id, tool_name, arguments, status, error_message, duration_ms, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        session_id,
                        tool_name,
                        json.dumps(arguments),
                        status,
                        error_message,
                        duration_ms,
                    ),
                )
            except Exception as trace_error:
                logger.warning(f"Failed to trace {tool_name}: {trace_error}")
    
    return wrapper

# Load embedding model once at startup
_embedding_model = None

def get_embedding_model():
    """Lazy-load the embedding model (expensive operation, only on first use)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

# Table names from environment variables
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME", "ticker_news_embeddings")
CHUNK_EMBEDDINGS_TABLE_NAME = os.environ.get("CHUNK_EMBEDDINGS_TABLE_NAME", "ticker_news_chunk_embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Context variable to store request headers for accessing end-user identity
_request_context: ContextVar[dict] = ContextVar('request_context', default={})


def _get_end_user_email() -> str:
    """Get the actual end user's email from request headers, or fallback to service principal."""
    # Try to get from X-Forwarded-User header (Databricks App context)
    headers = _request_context.get()
    forwarded_user = headers.get('x-forwarded-user')
    if forwarded_user:
        return forwarded_user
    
    # Fallback: use service principal (local development or non-App contexts)
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    return w.current_user.me().user_name or 'pricesetshedi@gmail.com'


mcp = FastMCP("alpaca-paper-trading")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to capture HTTP headers containing end-user identity."""
    async def dispatch(self, request: Request, call_next):
        # Capture headers that Databricks injects with user identity
        headers = {
            'x-forwarded-user': request.headers.get('x-forwarded-user'),
            'x-forwarded-email': request.headers.get('x-forwarded-email'),
        }
        _request_context.set(headers)
        response = await call_next(request)
        return response


@mcp.tool
@trace_mcp_call
def get_quote(symbol: str) -> dict:
    """
    Get the latest real quote for a stock ticker symbol from Massive.com.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        A dict with symbol, price, as_of (ISO timestamp), volume, change, and change_percent.
    """
    return massive_broker.get_quote(symbol)


@mcp.tool
@trace_mcp_call
def stage_trade(account_id: str, symbol: str, side: str, quantity: float) -> dict:
    """
    Stage a market order (BUY or SELL) for approval. Returns a confirmation
    code that must be passed to execute_trade to complete the trade.
    
    This two-step process ensures human-in-the-loop approval before executing
    real trades.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account (Alpaca paper trading is one account per
            API key pair).
        symbol: Stock ticker symbol, e.g. "AAPL".
        side: "BUY" or "SELL".
        quantity: Number of shares to trade (must be positive).

    Returns:
        A dict with trade summary, estimated cost, and confirmation_code.
    """
    # Validate inputs
    side = side.upper()
    if side not in ('BUY', 'SELL'):
        return {
            "status": "error",
            "message": "side must be 'BUY' or 'SELL'"
        }
    
    if quantity <= 0:
        return {
            "status": "error",
            "message": "quantity must be positive"
        }
    
    # Try to get quote for estimated cost (may fail for crypto)
    estimated_price = None
    estimated_cost = None
    try:
        quote = massive_broker.get_quote(symbol)
        estimated_price = quote.get("price")
        if estimated_price:
            estimated_cost = estimated_price * quantity
    except Exception as e:
        logger.info(f"Could not get quote for {symbol}: {e}")
    
    # Generate confirmation code
    confirmation_code = f"{random.randint(0, 99999):05d}"
    
    # Store pending trade
    _PENDING_TRADES[confirmation_code] = {
        "account_id": account_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "staged_at": time.time(),
    }
    
    return {
        "status": "staged",
        "message": f"Trade staged. Use execute_trade('{confirmation_code}') to confirm.",
        "confirmation_code": confirmation_code,
        "summary": {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "estimated_price": estimated_price,
            "estimated_cost": estimated_cost,
        },
    }


@mcp.tool
@trace_mcp_call
def execute_trade(confirmation_code: str) -> dict:
    """
    Execute a previously staged trade using its confirmation code.
    
    The confirmation code expires after 300 seconds. Once executed or expired,
    the code cannot be reused.

    Args:
        confirmation_code: The 5-digit code returned by stage_trade.

    Returns:
        A dict describing the executed order (id, symbol, side, quantity,
        price, notional, status, created_at).
    """
    # Check if code exists
    if confirmation_code not in _PENDING_TRADES:
        return {
            "status": "error",
            "message": f"Unknown or expired confirmation code: {confirmation_code}"
        }
    
    trade = _PENDING_TRADES[confirmation_code]
    
    # Check expiration (300 seconds)
    if time.time() - trade["staged_at"] > 300:
        del _PENDING_TRADES[confirmation_code]
        return {
            "status": "error",
            "message": "Confirmation code expired (300 second limit)"
        }
    
    # Delete code BEFORE executing to prevent double-execution on retry
    del _PENDING_TRADES[confirmation_code]
    
    # Execute the trade
    try:
        result = alpaca_broker.place_order(
            trade["account_id"],
            trade["symbol"],
            trade["side"],
            trade["quantity"],
        )
        result["confirmation_code"] = confirmation_code
        return result
    except Exception as e:
        logger.exception(f"Failed to execute trade {confirmation_code}")
        return {
            "status": "error",
            "message": f"Trade execution failed: {str(e)}",
            "confirmation_code": confirmation_code,
        }


@mcp.tool
@trace_mcp_call
def get_positions(account_id: str) -> list[dict]:
    """
    Get all open positions for the Alpaca paper trading account.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A list of dicts, each with symbol, quantity, avg_cost, updated_at.
    """
    return alpaca_broker.get_positions(account_id)


@mcp.tool
@trace_mcp_call
def get_account_summary(account_id: str) -> dict:
    """
    Get a full account summary for the Alpaca paper trading account: cash
    balance, open positions marked-to-market, total market value, and
    total equity (cash + market value).

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A dict with account_id, cash_balance, positions, market_value,
        total_equity.
    """
    return alpaca_broker.get_account_summary(account_id)


@mcp.tool
@trace_mcp_call
def get_order_history(account_id: str, limit: int = 50) -> list[dict]:
    """
    Get recent orders for the Alpaca paper trading account, most recent first.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.
        limit: Max number of orders to return (default 50).

    Returns:
        A list of dicts, each with id, symbol, side, quantity, price,
        notional, status, created_at.
    """
    return alpaca_broker.get_order_history(account_id, limit)


@mcp.tool
@trace_mcp_call
def get_balance(account_id: str) -> dict:
    """
    Get the current cash balance and buying power for the Alpaca paper 
    trading account.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A dict with account_id, cash_balance, buying_power, and currency.
    """
    return alpaca_broker.get_account_summary(account_id)


@mcp.tool
@trace_mcp_call
def get_current_user() -> dict:
    """
    Get information about the currently authenticated end user accessing the MCP server.
    
    When running as a Databricks App, this returns the actual end user making the
    request (from X-Forwarded-User header), not the service principal running the app.

    Returns:
        A dict with user_name (email from X-Forwarded-User header), 
        forwarded_email, and source ("request_header" or "service_principal").
    """
    try:
        # First, try to get the end user from the request headers
        # Databricks injects X-Forwarded-User with the actual user's email
        headers = _request_context.get()
        forwarded_user = headers.get('x-forwarded-user')
        forwarded_email = headers.get('x-forwarded-email')
        
        if forwarded_user:
            return {
                "status": "success",
                "user_name": forwarded_user,
                "forwarded_email": forwarded_email,
                "source": "request_header",
            }
        
        # Fallback: return the service principal if headers aren't available
        # (e.g., when running locally or in non-App contexts)
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        user = w.current_user.me()
        return {
            "status": "success",
            "user_name": user.user_name,
            "display_name": user.display_name,
            "active": user.active,
            "source": "service_principal",
        }
    except Exception as e:
        logger.exception("Failed to get current user")
        return {
            "status": "error",
            "message": f"Failed to get current user: {str(e)}",
        }


@mcp.tool
@trace_mcp_call
def add_to_watchlist(symbol: str, email: str = "pricesetshedi@gmail.com") -> dict:
    """
    Add a stock to the watchlist by fetching its current quote from Massive.com
    and storing it in the Lakebase watchlist table.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        email: User's email address (defaults to pricesetshedi@gmail.com).
    
    Returns:
        A dict with the quote data and confirmation that it was added to the watchlist.
    """
    try:
        # Use the provided email parameter
        user_email = email
        
        # Get quote from Massive.com
        quote = massive_broker.get_quote(symbol)
        
        # Store in Lakebase watchlist table
        sql = """
        INSERT INTO watchlist (email, symbol, latest_price, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (email, symbol) 
        DO UPDATE SET 
            latest_price = EXCLUDED.latest_price,
            updated_at = NOW()
        """
        
        lakebase.run_write(
            sql,
            (
                user_email,
                quote["symbol"],
                quote["price"]
            ),
        )
        
        return {
            "status": "success",
            "message": f"Added {symbol} to watchlist for {user_email}",
            "user_email": user_email,
            "quote": quote,
        }
    except Exception as e:
        logger.exception(f"Failed to add {symbol} to watchlist")
        return {
            "status": "error",
            "message": f"Failed to add {symbol} to watchlist: {str(e)}",
        }


@mcp.tool
@trace_mcp_call
def get_watchlist(limit: int = 100, email: str = 'pricesetshedi@gmail.com') -> dict:
    """
    Retrieve all stocks in the authenticated user's watchlist from Lakebase.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        limit: Maximum number of entries to return (default: 100).
        email: authenticate user's email
    
    Returns:
        A dict with watchlist entries sorted by most recently added.
    """
    try:
        # Get the actual end user's email (not the service principal)
        
        sql = """
        SELECT 
            symbol,
            latest_price,
            updated_at
        FROM watchlist
        WHERE email = %s
        LIMIT %s
        """
        
        rows = lakebase.run_query(sql, (email, limit))
        
        return {
            "status": "success",
            "user_email": email,
            "count": len(rows),
            "watchlist": rows,
        }
    except Exception as e:
        logger.exception(f"Failed to retrieve watchlist")
        return {
            "status": "error",
            "message": f"Failed to retrieve watchlist: {str(e)}",
        }


@mcp.tool
@trace_mcp_call
def remove_from_watchlist(symbol: str) -> dict:
    """
    Remove a stock from the authenticated user's watchlist.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        symbol: Stock ticker symbol to remove, e.g. "AAPL".
    
    Returns:
        A dict with status and confirmation message.
    """
    try:
        # Get the actual end user's email (not the service principal)
        user_email = _get_end_user_email()
        
        symbol = symbol.strip().upper()
        
        sql = """
        DELETE FROM watchlist
        WHERE email = %s AND symbol = %s
        """
        
        rows_affected = lakebase.run_write(sql, (user_email, symbol))
        
        if rows_affected > 0:
            return {
                "status": "success",
                "message": f"Removed {symbol} from watchlist",
                "symbol": symbol,
                "user_email": user_email,
            }
        else:
            return {
                "status": "not_found",
                "message": f"{symbol} was not in the watchlist",
                "symbol": symbol,
                "user_email": user_email,
            }
    except Exception as e:
        logger.exception(f"Failed to remove {symbol} from watchlist")
        return {
            "status": "error",
            "message": f"Failed to remove {symbol} from watchlist: {str(e)}",
        }


@mcp.tool
@trace_mcp_call
def get_stock_information(query: str, limit: int = 10, search_chunks: bool = True) -> dict:
    """
    Retrieve recent news and sentiment about a stock.
    
    Use this before making trading decisions to understand recent developments,
    market sentiment, and news about the stock. Searches ticker news using
    semantic similarity to find the most relevant information.
    
    Args:
        query: Stock symbol or natural language query (e.g. "AAPL", "tech company earnings")
        limit: Maximum number of results to return (default 10)
        search_chunks: Whether to search chunk-level embeddings in addition to documents
    
    Returns:
        A dict with query, documents, chunks, and model name
    """
    if not query or not query.strip():
        return {"error": "Query text is required"}
    
    try:
        # Compute embedding for the query
        model = get_embedding_model()
        query_embedding = model.encode(query)
        
        # Convert to list for JSON serialization and postgres array format
        embedding_list = query_embedding.tolist()
        
        # Search document-level embeddings
        doc_results = lakebase.run_query(
            f"""
            SELECT 
                e.id,
                e.ticker,
                e.title,
                e.published_utc,
                e.model_name,
                1 - (e.embedding <=> %s::vector) as similarity,
                d.description,
                d.article_url,
                d.sentiment
            FROM {EMBEDDINGS_TABLE_NAME} e
            LEFT JOIN {NEWS_TABLE_NAME} d ON e.id = d.id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (str(embedding_list), str(embedding_list), limit),
        )
        
        chunk_results = []
        if search_chunks:
            # Search chunk-level embeddings
            chunk_results = lakebase.run_query(
                f"""
                SELECT 
                    c.id,
                    c.article_id,
                    c.ticker,
                    c.chunk_index,
                    c.chunk_text,
                    c.model_name,
                    1 - (c.embedding <=> %s::vector) as similarity,
                    d.title,
                    d.article_url,
                    d.published_utc
                FROM {CHUNK_EMBEDDINGS_TABLE_NAME} c
                LEFT JOIN {NEWS_TABLE_NAME} d ON c.article_id = d.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (str(embedding_list), str(embedding_list), limit),
            )
        
        return {
            "query": query,
            "documents": doc_results,
            "chunks": chunk_results,
            "model": EMBEDDING_MODEL
        }
        
    except Exception as e:
        logger.exception("Vector search failed")
        return {"error": str(e)}


if __name__ == "__main__":
    # Add middleware to capture request headers for end-user identity
    # This must be done before mcp.run() is called
    if hasattr(mcp, 'app') and mcp.app is not None:
        mcp.app.add_middleware(RequestContextMiddleware)
    
    # Databricks Apps route external HTTP traffic to this port via app.yaml;
    # streamable-http is the transport Databricks' MCP client/gateway expects
    # (see the "Host your own MCP" doc linked in the module docstring above).
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)
