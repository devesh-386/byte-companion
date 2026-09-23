"""Windows' built-in acoustic echo cancellation (the Voice Capture DSP, CLSID_CWMAudioAEC).

    speakers ──► what Byte says ─┐
                                 ├──► Voice Capture DSP ──► mic audio with Byte's voice removed
    microphone ──► you + echo ───┘       (in Windows since Vista; Skype/Lync used it)

In "source mode" the DSP opens the default microphone itself and listens to the default speakers as
its reference, so it knows exactly what to subtract. We only pull clean 16 kHz mono audio out of it.

It's a DirectX Media Object (a COM object). No Python package wraps it, so the two interfaces we need
(IMediaObject, IPropertyStore) and the buffer it writes into (IMediaBuffer) are declared here by hand.
"""
import ctypes
import threading
import time
from ctypes import POINTER, byref, c_byte, c_long, c_ubyte, c_ulong, c_ushort, c_void_p, sizeof
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
CHUNK = 1600  # 0.1 s, the same block size the rest of voice.py uses


def available() -> bool:
    try:
        import comtypes  # noqa: F401
        return True
    except ImportError:
        return False


def _defs():
    """COM definitions, built lazily so importing this module never needs comtypes."""
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

    class IMediaBuffer(IUnknown):
        _iid_ = GUID("{59eff8b9-938c-4a26-82f2-95cb84cdc837}")
        _methods_ = [
            COMMETHOD([], HRESULT, "SetLength", (["in"], c_ulong, "cbLength")),
            COMMETHOD([], HRESULT, "GetMaxLength", (["out"], POINTER(c_ulong), "pcbMaxLength")),
            COMMETHOD([], HRESULT, "GetBufferAndLength", (["out"], POINTER(POINTER(c_ubyte)), "ppBuffer"),
                      (["out"], POINTER(c_ulong), "pcbLength")),
        ]

    class DMO_OUTPUT_DATA_BUFFER(ctypes.Structure):
        _fields_ = [("pBuffer", POINTER(IMediaBuffer)), ("dwStatus", c_ulong),
                    ("rtTimestamp", ctypes.c_longlong), ("rtTimelength", ctypes.c_longlong)]

    class WAVEFORMATEX(ctypes.Structure):
        _fields_ = [("wFormatTag", c_ushort), ("nChannels", c_ushort), ("nSamplesPerSec", c_ulong),
                    ("nAvgBytesPerSec", c_ulong), ("nBlockAlign", c_ushort), ("wBitsPerSample", c_ushort),
                    ("cbSize", c_ushort)]

    class DMO_MEDIA_TYPE(ctypes.Structure):
        _fields_ = [("majortype", GUID), ("subtype", GUID), ("bFixedSizeSamples", c_long),
                    ("bTemporalCompression", c_long), ("lSampleSize", c_ulong), ("formattype", GUID),
                    ("pUnk", c_void_p), ("cbFormat", c_ulong), ("pbFormat", c_void_p)]

    def unused(name):  # methods we never call still have to sit in the right vtable slot
        return COMMETHOD([], HRESULT, name)

    class IMediaObject(IUnknown):
        _iid_ = GUID("{d8ad0f58-5494-4102-97c5-ec798e59bcf4}")
        _methods_ = [
            unused("GetStreamCount"), unused("GetInputStreamInfo"), unused("GetOutputStreamInfo"),
            unused("GetInputType"), unused("GetOutputType"), unused("SetInputType"),
            COMMETHOD([], HRESULT, "SetOutputType", (["in"], c_ulong, "dwOutputStreamIndex"),
                      (["in"], POINTER(DMO_MEDIA_TYPE), "pmt"), (["in"], c_ulong, "dwFlags")),
            unused("GetInputCurrentType"), unused("GetOutputCurrentType"), unused("GetInputSizeInfo"),
            unused("GetOutputSizeInfo"), unused("GetInputMaxLatency"), unused("SetInputMaxLatency"),
            unused("Flush"), unused("Discontinuity"),
            COMMETHOD([], HRESULT, "AllocateStreamingResources"),
            COMMETHOD([], HRESULT, "FreeStreamingResources"),
            unused("GetInputStatus"), unused("ProcessInput"),
            COMMETHOD([], HRESULT, "ProcessOutput", (["in"], c_ulong, "dwFlags"),
                      (["in"], c_ulong, "cOutputBufferCount"),
                      (["in"], POINTER(DMO_OUTPUT_DATA_BUFFER), "pOutputBuffers"),
                      (["out"], POINTER(c_ulong), "pdwStatus")),
            unused("Lock"),
        ]

    class PROPERTYKEY(ctypes.Structure):
        _fields_ = [("fmtid", GUID), ("pid", c_ulong)]

    class PROPVARIANT(ctypes.Structure):
        # Only the two shapes we set: VT_I4 (a long) and VT_BOOL (a short). The union is 16 bytes on x64.
        _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort),
                    ("lVal", c_long), ("pad", c_byte * 12)]

    class IPropertyStore(IUnknown):
        _iid_ = GUID("{886d8eeb-8cf2-4446-8d02-cdba1dbdcf99}")
        _methods_ = [
            unused("GetCount"), unused("GetAt"), unused("GetValue"),
            COMMETHOD([], HRESULT, "SetValue", (["in"], POINTER(PROPERTYKEY), "key"),
                      (["in"], POINTER(PROPVARIANT), "propvar")),
            unused("Commit"),
        ]

    class MediaBuffer(comtypes.COMObject):
        """The buffer the DSP writes clean audio into. We own the memory."""
        _com_interfaces_ = [IMediaBuffer]

        def __init__(self, size: int):
            super().__init__()
            self.data = (c_ubyte * size)()
            self.size = size
            self.length = 0

        def IMediaBuffer_SetLength(self, this, n):
            self.length = n
            return 0

        def IMediaBuffer_GetMaxLength(self, this, pmax):
            pmax[0] = self.size
            return 0

        def IMediaBuffer_GetBufferAndLength(self, this, ppbuf, plen):
            if ppbuf:
                ppbuf[0] = ctypes.cast(self.data, POINTER(c_ubyte))
            if plen:
                plen[0] = self.length
            return 0

    return dict(IMediaObject=IMediaObject, IPropertyStore=IPropertyStore, PROPERTYKEY=PROPERTYKEY,
                PROPVARIANT=PROPVARIANT, DMO_MEDIA_TYPE=DMO_MEDIA_TYPE, WAVEFORMATEX=WAVEFORMATEX,
                DMO_OUTPUT_DATA_BUFFER=DMO_OUTPUT_DATA_BUFFER, MediaBuffer=MediaBuffer, GUID=GUID)


CLSID_AEC = "{745057c7-f353-4f2d-a7ee-58434477730e}"
AEC_PROPS = "{6f52c567-0360-4bd2-9617-ccbf1421c939}"   # MFPKEY_WMAAECMA_* property set
PID_SYSTEM_MODE = 2      # SINGLE_CHANNEL_AEC = 0
PID_SOURCE_MODE = 3      # TRUE: the DSP opens the mic and speakers itself
S_FALSE = 1
INCOMPLETE = 0x01000000  # DMO_OUTPUT_DATA_BUFFERF_INCOMPLETE: more audio is waiting


class EchoCancelledMic:
    """Streams echo-cancelled microphone audio as float32 blocks to `callback(block)`, like sounddevice.

    start() raises if Windows' AEC can't be opened; callers then fall back to the plain microphone."""

    def __init__(self, callback: Callable[[np.ndarray], None]):
        self.callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None

    def start(self, timeout: float = 5.0) -> None:
        self._stop.clear()
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="aec-mic")
        self._thread.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise RuntimeError("Windows echo cancellation did not start in time")
        if self._error:
            raise RuntimeError(f"Windows echo cancellation unavailable: {self._error}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(2)
        self._thread = None

    def _run(self) -> None:
        import comtypes
        import comtypes.client
        dmo = None
        inited = False
        try:
            d = _defs()
            try:
                comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
                inited = True
            except OSError:
                pass  # this thread already has COM set up (importing comtypes can do that)
            dmo = comtypes.client.CreateObject(d["GUID"](CLSID_AEC), interface=d["IMediaObject"])
            props = dmo.QueryInterface(d["IPropertyStore"])

            def set_prop(pid: int, vt: int, value: int) -> None:
                key = d["PROPERTYKEY"](d["GUID"](AEC_PROPS), pid)
                var = d["PROPVARIANT"]()
                var.vt, var.lVal = vt, value
                props.SetValue(byref(key), byref(var))

            set_prop(PID_SYSTEM_MODE, 3, 0)          # VT_I4, SINGLE_CHANNEL_AEC
            set_prop(PID_SOURCE_MODE, 11, -1)        # VT_BOOL, VARIANT_TRUE

            wfx = d["WAVEFORMATEX"](1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, 0)
            mt = d["DMO_MEDIA_TYPE"]()
            mt.majortype = d["GUID"]("{73647561-0000-0010-8000-00AA00389B71}")   # MEDIATYPE_Audio
            mt.subtype = d["GUID"]("{00000001-0000-0010-8000-00AA00389B71}")     # MEDIASUBTYPE_PCM
            mt.bFixedSizeSamples, mt.lSampleSize = 1, 2
            mt.formattype = d["GUID"]("{05589f81-c356-11ce-bf01-00aa0055595a}")  # FORMAT_WaveFormatEx
            mt.cbFormat, mt.pbFormat = sizeof(wfx), ctypes.cast(byref(wfx), c_void_p)
            dmo.SetOutputType(0, byref(mt), 0)
            dmo.AllocateStreamingResources()

            buf = d["MediaBuffer"](SAMPLE_RATE * 2)  # room for 1 s
            ibuf = buf.QueryInterface(_media_buffer_iface(d))
            out = d["DMO_OUTPUT_DATA_BUFFER"]()
            pending = np.zeros(0, dtype=np.float32)
            self._ready.set()
            while not self._stop.is_set():
                while True:
                    buf.length = 0
                    out.pBuffer, out.dwStatus = ibuf, 0
                    dmo.ProcessOutput(0, 1, byref(out))  # S_FALSE ("nothing yet") is a success code
                    if not buf.length:
                        break
                    pcm = np.frombuffer(bytes(buf.data[:buf.length]), dtype=np.int16)
                    pending = np.concatenate([pending, pcm.astype(np.float32) / 32768.0])
                    if not out.dwStatus & INCOMPLETE:
                        break
                while len(pending) >= CHUNK:
                    block, pending = pending[:CHUNK], pending[CHUNK:]
                    self.callback(block)
                time.sleep(0.01)
        except Exception as e:  # noqa: BLE001  reported to start(), which falls back to the plain mic
            self._error = e
            self._ready.set()
        finally:
            if dmo is not None:
                try:
                    dmo.FreeStreamingResources()
                except Exception:  # noqa: BLE001
                    pass
            if inited:
                comtypes.CoUninitialize()


def _media_buffer_iface(d):
    # IMediaBuffer is the only interface MediaBuffer implements.
    return d["MediaBuffer"]._com_interfaces_[0]
