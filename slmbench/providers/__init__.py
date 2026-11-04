"""Provider registry: maps a preset's ``kind`` to an implementation."""

from __future__ import annotations

from dataclasses import dataclass

from slmbench.config import ProviderPreset, get_preset, provider_presets, resolve_api_key
from slmbench.providers.base import Provider


class MissingAPIKey(RuntimeError):
    pass


def create_provider(provider_id: str, api_keys: dict[str, str] | None = None) -> Provider:
    preset = get_preset(provider_id)
    key = resolve_api_key(preset, api_keys)
    if preset.needs_key and not key:
        raise MissingAPIKey(f"{preset.name} needs {preset.api_key_env} (set it in .env or the sidebar)")
    if preset.kind == "openai_compat":
        from slmbench.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(preset, key)
    if preset.kind == "anthropic":
        from slmbench.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(preset, key)
    if preset.kind == "simulated":
        from slmbench.providers.simulated import SimulatedProvider

        return SimulatedProvider(preset)
    raise ValueError(f"No implementation for provider kind '{preset.kind}'")


@dataclass
class ProviderAvailability:
    preset: ProviderPreset
    has_key: bool

    @property
    def usable(self) -> bool:
        return self.has_key or not self.preset.needs_key


def remote_availability(api_keys: dict[str, str] | None = None) -> list[ProviderAvailability]:
    """Which hosted providers have credentials right now."""
    out = []
    for preset in provider_presets().values():
        if preset.local or preset.kind == "simulated" or preset.id == "custom":
            continue
        out.append(ProviderAvailability(preset, bool(resolve_api_key(preset, api_keys))))
    return out
