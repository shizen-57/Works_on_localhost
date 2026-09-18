from __future__ import annotations

import pytest

from app.config import ConfigError, load_settings


def test_sleepyai_configuration_is_supported():
    settings = load_settings(
        {
            "LLM_PROVIDER": "sleepyai",
            "LLM_MODEL": "example-model",
            "LLM_API_KEY": "test-only-key",
            "LLM_BASE_URL": "https://www.sleepyai.org/api/v1",
        }
    )
    assert settings.llm_provider == "sleepyai"
    assert settings.llm_base_url == "https://www.sleepyai.org/api/v1"


def test_sleepyai_requires_base_url():
    with pytest.raises(ConfigError, match="LLM_BASE_URL"):
        load_settings(
            {
                "LLM_PROVIDER": "sleepyai",
                "LLM_MODEL": "example-model",
                "LLM_API_KEY": "test-only-key",
            }
        )


def test_unimplemented_provider_is_rejected_during_config_parse():
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        load_settings(
            {
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": "example-model",
                "LLM_API_KEY": "test-only-key",
            }
        )


def test_invalid_log_level_is_rejected_during_config_parse():
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        load_settings(
            {
                "LLM_PROVIDER": "placeholder",
                "ALLOW_STUB_INTERPRETER": "true",
                "LOG_LEVEL": "very-chatty",
            }
        )


def test_placeholder_requires_explicit_dev_escape_hatch():
    with pytest.raises(ConfigError, match="ALLOW_STUB_INTERPRETER"):
        load_settings({"LLM_PROVIDER": "placeholder"})


@pytest.mark.parametrize("missing", ["LLM_MODEL", "LLM_API_KEY"])
def test_real_provider_requires_model_and_key(missing):
    env = {
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "model",
        "LLM_API_KEY": "key",
    }
    del env[missing]
    with pytest.raises(ConfigError, match=missing):
        load_settings(env)


@pytest.mark.parametrize("port", ["abc", "0", "65536"])
def test_invalid_port_is_rejected(port):
    with pytest.raises(ConfigError, match="PORT"):
        load_settings(
            {
                "LLM_PROVIDER": "placeholder",
                "ALLOW_STUB_INTERPRETER": "true",
                "PORT": port,
            }
        )


@pytest.mark.parametrize("deadline", ["abc", "0", "31"])
def test_invalid_deadline_is_rejected(deadline):
    with pytest.raises(ConfigError, match="REQUEST_DEADLINE_S"):
        load_settings(
            {
                "LLM_PROVIDER": "placeholder",
                "ALLOW_STUB_INTERPRETER": "true",
                "REQUEST_DEADLINE_S": deadline,
            }
        )


def test_placeholder_setting_reports_stub_usage():
    settings = load_settings(
        {
            "LLM_PROVIDER": "placeholder",
            "ALLOW_STUB_INTERPRETER": "true",
        }
    )
    assert settings.using_stub_interpreter is True
