from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigError(ValueError):
    """配置加载或校验异常。"""


class PeriodsPerYear:
    MINUTE_INTRADAY: int = 252 * 390
    HOUR_INTRADAY: int = 252 * 6
    DAILY: int = 252
    FOREX_INTRADAY_1M: int = 260 * 1380
    presets = {
        "minute_intraday": MINUTE_INTRADAY,
        "hour_intraday": HOUR_INTRADAY,
        "daily": DAILY,
        "forex_intraday_1m": FOREX_INTRADAY_1M,
    }


PresetPeriods = Literal["minute_intraday", "hour_intraday", "daily", "forex_intraday_1m"]


class RuntimeConfig(BaseModel):
    backend: Literal["auto", "cpu", "gpu"] = "auto"
    use_gpu: bool = True
    initial_cash: float = 1_000_000.0
    contract_multiplier: float = 1.0
    risk_free_rate: float = 0.02
    preset_periods: PresetPeriods | None = None
    periods_per_year: int = PeriodsPerYear.FOREX_INTRADAY_1M
    timezone: str = "Asia/Shanghai"
    default_interval: str = "1m"

    model_config = ConfigDict(extra="ignore")

    @field_validator("initial_cash", "contract_multiplier", "risk_free_rate")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("value must be >= 0")
        return float(value)

    @field_validator("periods_per_year")
    @classmethod
    def _positive_periods(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("periods_per_year must be > 0")
        return int(value)

    @model_validator(mode="after")
    @classmethod
    def _apply_period_preset(cls, values: RuntimeConfig) -> RuntimeConfig:
        if values.preset_periods is not None:
            preset_value = PeriodsPerYear.presets[values.preset_periods]
            values.periods_per_year = preset_value
        return values

    @field_validator("default_interval", "timezone")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        token = (value or "").strip()
        if not token:
            raise ValueError("string value cannot be empty")
        return token


class RulesConfig(BaseModel):
    margin_ratio: float = 0.1
    limit_up_ratio: float | None = None
    limit_down_ratio: float | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("margin_ratio")
    @classmethod
    def _positive_margin(cls, value: float) -> float:
        if value < 0:
            raise ValueError("margin_ratio must be >= 0")
        return float(value)

    @field_validator("limit_up_ratio", "limit_down_ratio")
    @classmethod
    def _ratio_range(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value < 0:
            raise ValueError("limit ratio must be >= 0")
        return float(value)


class SlippageConfig(BaseModel):
    model: Literal["fixed", "percent", "volume"] = "percent"
    value: float = 0.0001
    impact: float = 1.0

    model_config = ConfigDict(extra="ignore")

    @field_validator("value", "impact")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("slippage value must be >= 0")
        return float(value)


class CommissionConfig(BaseModel):
    model: Literal["fixed", "percent", "tiered"] = "percent"
    value: float = 0.0002
    tiers: list[tuple[float, float]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("value")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("commission value must be >= 0")
        return float(value)


class OptimizeConfig(BaseModel):
    metric: str = "sharpe"
    maximize: bool = True
    n_trials: int = 200
    max_workers: int | None = None
    random_seed: int = 42
    batch_size: int = 128

    model_config = ConfigDict(extra="ignore")

    @field_validator("n_trials", "batch_size")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be > 0")
        return int(value)

    @field_validator("random_seed")
    @classmethod
    def _non_negative_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_seed must be >= 0")
        return int(value)


class VisualizationConfig(BaseModel):
    theme: str = "plotly_dark"
    width: int = 1280
    height: int = 720
    include_trade_table: bool = True

    model_config = ConfigDict(extra="ignore")

    @field_validator("theme")
    @classmethod
    def _theme(cls, value: str) -> str:
        token = (value or "").strip()
        if not token:
            raise ValueError("theme cannot be empty")
        return token

    @field_validator("width", "height")
    @classmethod
    def _min_size(cls, value: int) -> int:
        if value < 300:
            raise ValueError("width and height must be >= 300")
        return int(value)


class QuantEngineConfig(BaseModel):
    data_root: str = "./data"
    results_root: str = "./results"
    reports_root: str = "./reports"
    runtime: RuntimeConfig = RuntimeConfig()
    slippage: SlippageConfig = SlippageConfig()
    commission: CommissionConfig = CommissionConfig()
    optimize: OptimizeConfig = OptimizeConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    rules: RulesConfig = RulesConfig()

    model_config = ConfigDict(extra="ignore")

    @field_validator("data_root", "results_root", "reports_root")
    @classmethod
    def _non_empty_path(cls, value: str) -> str:
        token = (value or "").strip()
        if not token:
            raise ValueError("path cannot be empty")
        return token


def load_config(path: str | Path | None) -> QuantEngineConfig:
    if path is None:
        return QuantEngineConfig()
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigError(f"配置文件不存在: {file_path}")
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        return QuantEngineConfig.model_validate(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"读取配置失败: {exc}") from exc
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
