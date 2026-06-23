import asyncio
from typing import Callable, Awaitable
from events.event_types import Event

class EventBus:
    def __init__(self):
        self._subscribers: list[Callable[[Event], Awaitable[None]]] = []
        
    def subscribe(self, callback: Callable[[Event], Awaitable[None]]):
        self._subscribers.append(callback)
        
    async def publish(self, event: Event):
        for sub in self._subscribers:
            try:
                await sub(event)
            except Exception as e:
                print(f"Error in subscriber: {e}")

# Global instance
bus = EventBus()
