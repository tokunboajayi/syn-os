"""
Device Management — /api/v1/devices
Full CRUD for registered network devices.
"""
from __future__ import annotations

import uuid
import time
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/devices", tags=["devices"])

# ── Models ─────────────────────────────────────────────────────────────────────
class DeviceCreate(BaseModel):
    name: str
    ip: str
    type: str = "generic"   # workstation | server | iot | mobile | generic
    label: Optional[str] = None
    meta: Optional[Dict] = None

class DeviceUpdate(BaseModel):
    name: Optional[str] = None
    label: Optional[str] = None
    meta: Optional[Dict] = None

class Device(BaseModel):
    id: str
    name: str
    ip: str
    type: str
    label: Optional[str]
    status: str              # online | offline | unknown
    last_seen: Optional[float]
    registered_at: float
    meta: Optional[Dict]

# ── In-memory store ────────────────────────────────────────────────────────────
_DEVICES: Dict[str, Device] = {}

# Seed with a couple of demo devices so the UI isn't empty on first load
def _seed():
    demos = [
        DeviceCreate(name="Gateway Router", ip="192.168.1.1", type="server", label="NAT Gateway"),
        DeviceCreate(name="Workstation-01", ip="192.168.1.101", type="workstation"),
        DeviceCreate(name="Raspberry Pi Sensor", ip="192.168.1.55", type="iot", label="Env sensor"),
    ]
    statuses = ["online", "online", "offline"]
    for d, s in zip(demos, statuses):
        dev_id = str(uuid.uuid4())
        _DEVICES[dev_id] = Device(
            id=dev_id,
            name=d.name, ip=d.ip, type=d.type, label=d.label,
            status=s, last_seen=time.time(), registered_at=time.time(), meta=d.meta,
        )

_seed()

# ── Helpers ────────────────────────────────────────────────────────────────────
async def _ping(ip: str) -> bool:
    import asyncio, subprocess
    try:
        proc = await asyncio.create_subprocess_exec(
            "ping", "-n", "1", "-w", "800", ip,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=2)
        return proc.returncode == 0
    except Exception:
        return False

# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.get("", response_model=List[Device])
async def list_devices():
    return list(_DEVICES.values())


@router.post("", response_model=Device, status_code=201)
async def register_device(body: DeviceCreate):
    dev_id = str(uuid.uuid4())
    device = Device(
        id=dev_id,
        name=body.name, ip=body.ip, type=body.type,
        label=body.label, status="unknown",
        last_seen=None, registered_at=time.time(), meta=body.meta,
    )
    _DEVICES[dev_id] = device
    return device


@router.get("/{device_id}", response_model=Device)
async def get_device(device_id: str):
    device = _DEVICES.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


@router.patch("/{device_id}", response_model=Device)
async def update_device(device_id: str, body: DeviceUpdate):
    device = _DEVICES.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    data = device.dict()
    if body.name is not None:
        data["name"] = body.name
    if body.label is not None:
        data["label"] = body.label
    if body.meta is not None:
        data["meta"] = body.meta
    _DEVICES[device_id] = Device(**data)
    return _DEVICES[device_id]


@router.delete("/{device_id}", status_code=204)
async def delete_device(device_id: str):
    if device_id not in _DEVICES:
        raise HTTPException(status_code=404, detail="Device not found")
    del _DEVICES[device_id]


@router.post("/{device_id}/ping")
async def ping_device(device_id: str):
    """Ping the device and update its online status."""
    device = _DEVICES.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    online = await _ping(device.ip)
    data = device.dict()
    data["status"] = "online" if online else "offline"
    data["last_seen"] = time.time() if online else data["last_seen"]
    _DEVICES[device_id] = Device(**data)
    return {"id": device_id, "status": data["status"]}
