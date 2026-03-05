# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""Kalshi data client for NautilusTrader.

Provides instrument loading from the Kalshi API. Market data (quotes) is
minimal since the weather strategy generates its own signals from forecast APIs.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from nautilus_trader.adapters.kalshi.common.constants import KALSHI_VENUE
from nautilus_trader.live.data_client import LiveDataClient
from nautilus_trader.model.identifiers import ClientId

if TYPE_CHECKING:
    from nautilus_trader.adapters.kalshi.config import KalshiDataClientConfig
    from nautilus_trader.adapters.kalshi.http.client import KalshiHttpClient
    from nautilus_trader.adapters.kalshi.providers import KalshiInstrumentProvider
    from nautilus_trader.cache.cache import Cache
    from nautilus_trader.common.component import LiveClock
    from nautilus_trader.common.component import MessageBus


class KalshiDataClient(LiveDataClient):
    """
    Provides a data client for the Kalshi prediction market.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The event loop for the client.
    http_client : KalshiHttpClient
        The Kalshi HTTP client.
    msgbus : MessageBus
        The message bus for the client.
    cache : Cache
        The cache for the client.
    clock : LiveClock
        The clock for the client.
    instrument_provider : KalshiInstrumentProvider
        The instrument provider for the client.
    config : KalshiDataClientConfig
        The configuration for the client.
    name : str, optional
        The custom client name.

    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        http_client: KalshiHttpClient,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
        instrument_provider: KalshiInstrumentProvider,
        config: KalshiDataClientConfig,
        name: str | None = None,
    ):
        super().__init__(
            loop=loop,
            client_id=ClientId(name or KALSHI_VENUE.value),
            venue=KALSHI_VENUE,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        self._instrument_provider = instrument_provider
        self._http_client = http_client
        self._config = config

    async def _connect(self) -> None:
        """Connect to the Kalshi data feeds — load instruments."""
        await self._instrument_provider.load_all_async()
        self._log.info(
            f"Loaded {len(self._instrument_provider.list_all())} instruments",
        )

    async def _disconnect(self) -> None:
        """Disconnect from Kalshi data feeds."""
        self._log.info("Kalshi data client disconnected")

    async def _subscribe_quote_ticks(self, instrument_id) -> None:
        """Subscribe to quote ticks — not implemented (strategy generates its own signals)."""
        self._log.warning(
            f"Quote tick subscription not implemented for Kalshi: {instrument_id}",
        )

    async def _unsubscribe_quote_ticks(self, instrument_id) -> None:
        """Unsubscribe from quote ticks."""
