from __future__ import annotations

from dataclasses import dataclass

from .schemas import PermittedUse, ProviderProfile, RunConfig, RunMode


class MissingProviderProfileError(LookupError):
    """Raised when an adapter tries to operate without a registered profile."""


class RegistryNotBoundError(RuntimeError):
    """Raised when `gate()` is invoked before `bind()` provides a run config."""


@dataclass(frozen=True)
class GateResult:
    """The outcome of a per-call compliance gate evaluation.

    `ok` is True only when the provider is permitted to operate for the active
    run configuration; otherwise `reason` carries a stable, machine-readable
    string suitable for the audit ledger and validator.
    """

    ok: bool
    reason: str | None = None


def _research_only_blocked_in(run_mode: RunMode) -> bool:
    """`research_only` providers may not be cited in distributed (shared) runs."""

    return run_mode is RunMode.distributed


def _dev_only_blocked_in(run_mode: RunMode) -> bool:
    """`dev_only` providers may not be cited in distributed (shared) runs.

    They are permitted in personal_research runs since the user is the only
    consumer of the output.
    """

    return run_mode is RunMode.distributed


class ProviderRegistry:
    """Holds provider compliance profiles and gates them at use time.

    Two-step lifecycle:
      1. `register(profile)` once at startup for every known provider.
      2. `bind(run_config)` captures an immutable run-time snapshot.
    `gate(profile_id)` is then invoked for every adapter call and resolves
    against that snapshot, never against mutable state.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ProviderProfile] = {}
        self._run_config: RunConfig | None = None

    def register(self, profile: ProviderProfile) -> None:
        if profile.id in self._profiles:
            raise ValueError(f"duplicate provider profile id: {profile.id!r}")
        self._profiles[profile.id] = profile

    def register_many(self, profiles: list[ProviderProfile]) -> None:
        for p in profiles:
            self.register(p)

    def get(self, profile_id: str) -> ProviderProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as e:
            raise MissingProviderProfileError(
                f"no provider profile registered for id={profile_id!r}"
            ) from e

    def ids(self) -> list[str]:
        return sorted(self._profiles.keys())

    def bind(self, run_config: RunConfig) -> "ProviderRegistry":
        """Capture the immutable run-config snapshot. Returns self for chaining."""

        if self._run_config is not None:
            raise RuntimeError(
                "ProviderRegistry already bound; create a fresh registry per run"
            )
        self._run_config = run_config
        return self

    def is_bound(self) -> bool:
        return self._run_config is not None

    def gate(self, profile_id: str) -> GateResult:
        """Per-call compliance gate. Invoked at every adapter call."""

        if self._run_config is None:
            raise RegistryNotBoundError(
                "ProviderRegistry.gate() called before bind(run_config)"
            )
        try:
            profile = self._profiles[profile_id]
        except KeyError as e:
            raise MissingProviderProfileError(
                f"no provider profile registered for id={profile_id!r}"
            ) from e

        run_mode = self._run_config.run_mode

        if profile.permitted_use is PermittedUse.research_only and _research_only_blocked_in(run_mode):
            return GateResult(
                ok=False,
                reason=(
                    f"profile {profile.id!r} is research_only; cannot be used in "
                    f"run_mode={run_mode.value}"
                ),
            )
        if profile.permitted_use is PermittedUse.dev_only and _dev_only_blocked_in(run_mode):
            return GateResult(
                ok=False,
                reason=(
                    f"profile {profile.id!r} is dev_only; cannot be used in "
                    f"run_mode={run_mode.value}"
                ),
            )
        if profile.entitlement_required:
            entitled = False
            if profile.id.startswith("moomoo"):
                entitled = bool(self._run_config.moomoo_entitled)
            if not entitled:
                return GateResult(
                    ok=False,
                    reason=(
                        f"profile {profile.id!r} requires user entitlement; "
                        "run_config has not confirmed it"
                    ),
                )

        return GateResult(ok=True)

    def profile_versions(self) -> dict[str, str]:
        """Snapshot of {profile_id: profile_version} for the audit ledger."""

        return {pid: p.profile_version for pid, p in self._profiles.items()}
