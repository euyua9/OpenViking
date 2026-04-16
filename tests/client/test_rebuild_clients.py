from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import openviking_cli.client.http as http_module
from openviking import AsyncOpenViking, SyncOpenViking
from openviking.client.local import LocalClient
from openviking_cli.client.http import AsyncHTTPClient
from openviking_cli.client.sync_http import SyncHTTPClient
from openviking_cli.utils.config import OPENVIKING_CLI_CONFIG_ENV


@pytest.fixture(autouse=True)
def clear_ovcli_config(monkeypatch):
    monkeypatch.delenv(OPENVIKING_CLI_CONFIG_ENV, raising=False)
    monkeypatch.setattr(http_module, "load_ovcli_config", lambda: None)


async def test_async_openviking_rebuild_forwards_to_local_client(tmp_path):
    client = AsyncOpenViking(path=str(tmp_path))
    with patch.object(client, "_ensure_initialized", new_callable=AsyncMock) as mock_init:
        with patch.object(client._client, "rebuild", new_callable=AsyncMock) as mock_rebuild:
            mock_rebuild.return_value = {"status": "completed"}

            result = await client.rebuild(
                "viking://resources/demo",
                mode="vectors_only",
                wait=False,
                reason="ops",
            )

    assert result == {"status": "completed"}
    mock_init.assert_awaited_once()
    mock_rebuild.assert_awaited_once_with(
        uri="viking://resources/demo",
        mode="vectors_only",
        wait=False,
        reason="ops",
    )


def test_sync_openviking_rebuild_forwards_to_async_client():
    client = SyncOpenViking()
    with patch.object(
        client._async_client,
        "rebuild",
        return_value={"status": "completed"},
    ) as mock_rebuild:
        with patch(
            "openviking.sync_client.run_async", return_value={"status": "completed"}
        ) as mock_run:
            result = client.rebuild(
                "viking://resources/demo",
                mode="semantic_and_vectors",
                wait=True,
                reason="ops",
            )

    assert result == {"status": "completed"}
    assert mock_run.called
    assert mock_rebuild.called


async def test_local_client_rebuild_forwards_to_service():
    client = LocalClient.__new__(LocalClient)
    client._service = SimpleNamespace(rebuild=AsyncMock(return_value={"status": "completed"}))

    result = await LocalClient.rebuild(
        client,
        uri="viking://resources/demo",
        mode="vectors_only",
        wait=False,
        reason="ops",
    )

    assert result == {"status": "completed"}
    client._service.rebuild.assert_awaited_once()


async def test_async_http_client_rebuild_posts_content_rebuild():
    client = AsyncHTTPClient(url="http://localhost:1933")
    fake_http = SimpleNamespace(post=AsyncMock(return_value=object()))
    client._http = fake_http
    with patch.object(
        client, "_handle_response", return_value={"status": "completed"}
    ) as mock_handle:
        result = await client.rebuild(
            "viking://resources/demo",
            mode="vectors_only",
            wait=False,
            reason="ops",
        )

    assert result == {"status": "completed"}
    fake_http.post.assert_awaited_once_with(
        "/api/v1/content/rebuild",
        json={
            "uri": "viking://resources/demo",
            "mode": "vectors_only",
            "wait": False,
            "reason": "ops",
        },
    )
    assert mock_handle.called


def test_sync_http_client_rebuild_forwards_to_async_client():
    client = SyncHTTPClient(url="http://localhost:1933")
    with patch.object(
        client._async_client,
        "rebuild",
        return_value={"status": "accepted"},
    ) as mock_rebuild:
        with patch(
            "openviking_cli.client.sync_http.run_async",
            return_value={"status": "accepted"},
        ) as mock_run:
            result = client.rebuild(
                "viking://resources/demo",
                mode="vectors_only",
                wait=False,
                reason="ops",
            )

    assert result == {"status": "accepted"}
    assert mock_run.called
    assert mock_rebuild.called
