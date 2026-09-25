from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_robot_service_manager_supports_optional_camera_and_health() -> None:
    script = (ROOT / "systemd" / "configure-pico-services.sh").read_text()
    assert "health)" in script
    assert 'camera_enabled && units+=("${camera_unit}")' in script
    assert "report_listener_health udp" in script
    assert "report_listener_health tcp" in script
    assert "conflicting controller is enabled or active" in script


def test_robot_services_retry_without_systemd_start_limit() -> None:
    runtime = (ROOT / "systemd" / "microban-pico-runtime.service.in").read_text()
    camera = (ROOT / "systemd" / "microban-camera-tls.service.in").read_text()
    for unit in (runtime, camera):
        assert "StartLimitIntervalSec=0" in unit
        assert "StartLimitBurst=" not in unit

    assert "Wants=network-online.target microban-camera.service" in camera
    assert "Requires=microban-camera.service" not in camera


def test_robot_autostart_documentation_includes_health_action() -> None:
    documentation = (ROOT / "docs" / "pico_autostart.md").read_text()
    assert "configure-pico-services.sh health" in documentation
    assert "UDP 5555" in documentation
    assert "TLS 8443" in documentation
