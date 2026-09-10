"""
A cuFFT PlanNd-like wrapper with a caller-managed work area.

This module keeps CuPy as the array/memory layer and uses the low-level
nvmath.bindings.cufft API for plan creation and execution by default.
Pass use_cupy=True to delegate to CuPy's PlanNd with private workspace.

The important difference from cupy.cuda.cufft.PlanNd is that plan
creation and work-area allocation are separate operations. Several plans can
therefore be created first and then bound to one allocation whose size is the
maximum of their individual requirements, which allows for significant memory 
savings when multiple plans are used in a single simulation.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any
import operator

import cupy as cp
from nvmath.bindings import cufft as nvcufft

# cuFFT direction constants used by complex transforms.
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

# Limits used to select the matching cuFFT planning interface.
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1

# Convert nvmath enum values once so they can be compared with the integer
# fft_type values also accepted by CuPy's PlanNd interface.
_C2C = int(nvcufft.Type.C2C)
_R2C = int(nvcufft.Type.R2C)
_C2R = int(nvcufft.Type.C2R)
_Z2Z = int(nvcufft.Type.Z2Z)
_D2Z = int(nvcufft.Type.D2Z)
_Z2D = int(nvcufft.Type.Z2D)

_SUPPORTED_TYPES = {_C2C, _R2C, _C2R, _Z2Z, _D2Z, _Z2D}
_TYPE_NAMES = {
    _C2C: "C2C",
    _R2C: "R2C",
    _C2R: "C2R",
    _Z2Z: "Z2Z",
    _D2Z: "D2Z",
    _Z2D: "Z2D",
}


class FlucsPlanNd:
    """
    An N-dimensional, batched cuFFT plan with an external work area.

    Parameters are the same advanced-layout parameters used by
    cupy.cuda.cufft.PlanNd. The parameters order, last_axis, and last_size
    are retained as compatibility metadata; cuFFT's cufftMakePlanMany does
    not consume them.

    Parameters
    ----------
    shape, inembed, istride, idist, onembed, ostride, odist, fft_type, batch
        Parameters forwarded to cufftMakePlanMany (or its 64-bit form).
    order, last_axis, last_size
        CuPy-compatible plan metadata.
    work_area
        An optional workspace object or raw integer device pointer.  Accepted
        objects include cupy.cuda.Memory, cupy.cuda.MemoryPointer, and
        contiguous CuPy arrays.  A raw pointer requires work_area_size.
    work_area_size
        Number of accessible bytes starting at work_area.  It is inferred
        for supported CuPy objects and mandatory for a raw pointer.
    work_area_owner
        Optional object retained to keep a raw-pointer allocation alive.
    auto_allocate
        If true, allocate and bind a workspace using CuPy's current allocator.
        It cannot be combined with work_area.  The default is false so
        several plans can be constructed before allocating shared storage.
        Applies only to the custom backend; CuPy always allocates privately.
    use_64bit
        Select cufftMakePlanMany64.  If None, it is selected whenever
        an advanced-layout integer is outside the signed 32-bit range.
    use_cupy
        If true, wrap a cupy.cuda.cufft.PlanNd instead of creating a custom
        plan. External workspace arguments and 64-bit planning are unsupported.
        work_size reports CuPy's allocated capacity, which may include
        allocator rounding, rather than the exact cuFFT workspace requirement.

    Notes
    -----
    A work area may be shared only by FFT executions that cannot overlap on
    the GPU.  Enqueuing all such FFTs on one CUDA stream is sufficient.  If
    different streams are used, the caller must establish event dependencies
    that prevent overlap.

    The caller must keep arrays and streams alive until execution completes,
    and finish queued FFTs before rebinding workspace or closing a plan.
    Calls on a given plan must be serialized by the host as well. This class
    is not thread-safe. Both backends require CUFFT_FORWARD for R2C/D2Z and
    CUFFT_INVERSE for C2R/Z2D, rejecting directions cuFFT would silently ignore.

    Padded in-place real transforms require explicit embeddings in real/complex 
    element units respectively.
    """

    def __init__(
        self,
        shape: Iterable[int],
        inembed: Iterable[int] | None,
        istride: int,
        idist: int,
        onembed: Iterable[int] | None,
        ostride: int,
        odist: int,
        fft_type: int,
        batch: int,
        order: str = "C",
        last_axis: int = -1,
        last_size: int | None = None,
        *,
        work_area: Any | int | None = None,
        work_area_size: int | None = None,
        work_area_owner: Any | None = None,
        auto_allocate: bool = False,
        use_64bit: bool | None = None,
        use_cupy: bool = False,
    ) -> None:
        # Initialise resource state before any operation that may fail. This
        # allows the exception cleanup path to safely handle partial setup.
        self.handle = 0
        self._cupy_plan = None
        self._use_cupy = bool(use_cupy)
        self.gpus = None
        self.device_id = int(cp.cuda.runtime.getDevice())

        # Materialise and validate the advanced layout. In particular, this
        # prevents one-shot iterables from being consumed more than once.
        self.shape = _positive_int_tuple("shape", shape)
        self.inembed = _optional_positive_int_tuple(
            "inembed", inembed, len(self.shape)
        )
        self.onembed = _optional_positive_int_tuple(
            "onembed", onembed, len(self.shape)
        )
        self.istride = _positive_int("istride", istride)
        self.idist = operator.index(idist)
        self.ostride = _positive_int("ostride", ostride)
        self.odist = operator.index(odist)
        self.fft_type = int(fft_type)
        self.batch = operator.index(batch)
        self.order = str(order)
        self.last_axis = int(last_axis)
        self.last_size = None if last_size is None else int(last_size)

        # Check the relationships between layout, backend, and workspace
        # options before creating a GPU resource.
        if self.fft_type not in _SUPPORTED_TYPES:
            raise ValueError(f"unsupported cuFFT type: {fft_type!r}")
        if self.batch < 0:
            raise ValueError("batch must be non-negative")
        if self.idist < 0 or self.odist < 0:
            raise ValueError("batch distances must be non-negative")
        if self.order not in {"C", "F"}:
            raise ValueError("order must be 'C' or 'F'")
        if self.last_size is not None and self.last_size < 1:
            raise ValueError("last_size must be positive or None")
        if self.use_cupy and any(value is not None for value in (
            work_area, work_area_size, work_area_owner
        )):
            raise ValueError("use_cupy=True does not support external workspace arguments")
        if auto_allocate and work_area is not None:
            raise ValueError("auto_allocate and work_area are mutually exclusive")
        if work_area is None and work_area_size is not None:
            raise ValueError("work_area_size was supplied without work_area")
        if work_area is None and work_area_owner is not None:
            raise ValueError("work_area_owner was supplied without work_area")

        # The first nine entries match CuPy's backend PlanNd arguments.  The
        # final three reproduce the metadata historically present in CuPy's
        # cache key and in older PlanNd call sites.
        self.backend_plan_key = (
            self.shape,
            self.inembed,
            self.istride,
            self.idist,
            self.onembed,
            self.ostride,
            self.odist,
            self.fft_type,
            self.batch,
        )
        self.plan_key = self.backend_plan_key + (
            self.order,
            self.last_axis,
            self.last_size,
        )

        # Use the 64-bit planning API only when requested or required by an
        # advanced-layout value. The transform data type is unaffected.
        layout_ints = (
            *self.shape,
            *(self.inembed or ()),
            self.istride,
            self.idist,
            *(self.onembed or ()),
            self.ostride,
            self.odist,
            self.batch,
        )
        self.use_64bit = (
            any(x < _INT32_MIN or x > _INT32_MAX for x in layout_ints)
            if use_64bit is None
            else bool(use_64bit)
        )
        if not self.use_64bit and any(x > _INT32_MAX for x in layout_ints):
            raise ValueError(
                "layout exceeds signed 32-bit range; use use_64bit=True"
            )
        if any(x > (1 << 63) - 1 for x in layout_ints):
            raise ValueError("layout exceeds signed 64-bit range")
        if self.use_cupy and self.use_64bit:
            raise ValueError("use_cupy=True does not support 64-bit planning")

        # Track both the pointer passed to cuFFT and a Python owner that keeps
        # the underlying device allocation alive for the plan's lifetime.
        self.work_size = 0
        self.work_area: Any | int | None = None
        self.work_area_ptr = 0
        self.work_area_size = 0
        self._work_area_owner: Any | None = None
        self._work_area_bound = self.batch == 0

        try:
            if self.use_cupy:
                # CuPy creates the plan and privately allocates its workspace.
                # Retain the PlanNd object because it owns both resources.
                self._cupy_plan = cp.cuda.cufft.PlanNd(*self.plan_key)
                self.handle = int(self._cupy_plan.handle)
                self.work_area = self._cupy_plan.work_area
                if self.work_area is not None:
                    self.work_area_ptr, self.work_area_size, self._work_area_owner = (
                        _workspace_pointer_and_size(self.work_area, None)
                    )
                self.work_size = self.work_area_size
                self._work_area_bound = True
                return

            # Disable cuFFT's automatic allocation so planning reports the
            # required size without reserving a private workspace.
            self.handle = int(nvcufft.create())
            nvcufft.set_auto_allocation(self.handle, 0)

            if self.batch != 0:
                # cufftMakePlanMany configures the handle and returns the exact
                # temporary storage required by this layout.
                make_plan = (
                    nvcufft.make_plan_many64
                    if self.use_64bit
                    else nvcufft.make_plan_many
                )
                self.work_size = int(
                    make_plan(
                        self.handle,
                        len(self.shape),
                        list(self.shape),
                        0 if self.inembed is None else list(self.inembed),
                        self.istride,
                        self.idist,
                        0 if self.onembed is None else list(self.onembed),
                        self.ostride,
                        self.odist,
                        self.fft_type,
                        self.batch,
                    )
                )
                if self.work_size == 0:
                    self._work_area_bound = True

            # Workspace binding is deliberately separate from plan creation,
            # allowing the caller to share one allocation between plans.
            if work_area is not None:
                self.set_work_area(
                    work_area,
                    work_area_size,
                    owner=work_area_owner,
                )
            elif auto_allocate:
                self.allocate_work_area()
        except BaseException:
            # Do not leak a handle if validation or backend setup fails after
            # cufftCreate has succeeded.
            self._destroy_after_failed_init()
            raise

    @property
    def use_cupy(self) -> bool:
        """
        Whether this wrapper delegates to CuPy's privately allocated plan.
        """
        return self._use_cupy

    @property
    def closed(self) -> bool:
        """
        Whether the underlying cuFFT handle has been destroyed.
        """
        return self.handle == 0

    @property
    def required_work_area_size(self) -> int:
        """
        cuFFT workspace requirement, or allocated capacity in CuPy mode.
        """
        return self.work_size

    def set_work_area(
        self,
        work_area: Any | int,
        work_area_size: int | None = None,
        *,
        owner: Any | None = None,
    ) -> None:
        """
        Bind caller-owned device memory as this plan's work area.

        The caller must finish earlier FFTs before rebinding.
        When work_area is a raw integer pointer, the caller is responsible
        for keeping the allocation alive; passing owner lets this object
        retain that lifetime reference.
        """

        self._require_open()
        if self.use_cupy:
            raise RuntimeError(
                "CuPy manages this plan's workspace; it cannot be rebound"
            )

        # Accept raw pointers, CuPy memory objects, and contiguous arrays while
        # reducing them to the pointer and accessible capacity cuFFT needs.
        ptr, capacity, retained_object = _workspace_pointer_and_size(
            work_area, work_area_size
        )

        # Check capacity before binding; cuFFT receives only a pointer and
        # cannot verify that the allocation is large enough.
        if capacity < self.work_size:
            raise ValueError(
                f"Work area has {capacity} bytes, but this plan requires "
                f"{self.work_size} bytes"
            )
        if self.work_size and ptr == 0:
            raise ValueError("A non-zero work-area pointer is required")

        workspace_device = _object_device_id(retained_object)
        owner_device = _object_device_id(owner)

        # A pointer from another device may be numerically valid but is not a
        # usable workspace for this plan.
        for candidate in (workspace_device, owner_device):
            if candidate is not None and candidate != self.device_id:
                raise ValueError(
                    f"Work area is on device {candidate}, but the plan is on "
                    f"device {self.device_id}"
                )

        if self.batch != 0:
            nvcufft.set_work_area(self.handle, ptr)

        # Update lifetime references only after cufftSetWorkArea succeeds.
        self.work_area = work_area
        self.work_area_ptr = ptr
        self.work_area_size = capacity
        self._work_area_owner = owner if owner is not None else retained_object
        self._work_area_bound = True

    def allocate_work_area(
        self,
        allocator: Callable[[int], Any] | None = None,
    ) -> Any | None:
        """
        Allocate and bind a private work area, returning its owner object.

        The default is cupy.cuda.alloc, matching CuPy's use of the current
        allocator. Pass cupy.cuda.Memory to force a direct device-memory
        allocation outside the configured CuPy memory pool. Finish earlier FFTs 
        before replacing an existing work area.
        """

        self._require_open()
        if self.use_cupy:
            raise RuntimeError(
                "CuPy has already allocated this plan's private workspace"
            )
        if self.work_size == 0:
            self._work_area_bound = True
            return None

        # Allocate on the device where the plan was created, irrespective of
        # the device that happens to be current at the call site.
        if allocator is None:
            allocator = cp.cuda.alloc
        with cp.cuda.Device(self.device_id):
            work_area = allocator(self.work_size)
        self.set_work_area(work_area)
        return work_area

    def fft(self, a: cp.ndarray, out: cp.ndarray, direction: int) -> None:
        """
        Execute the plan on CuPy arrays using the current CUDA stream.

        Like CuPy's low-level PlanNd.fft, this operation is unnormalised and 
        enqueues work asynchronously. The caller supplies arrays with the
        correct dtype, device, layout and allocation size for the plan. The 
        caller manages array/stream lifetimes and serializes executions that use 
        the same plan or shared workspace. No events or waits are inserted by 
        this wrapper.
        """

        self._require_open()
        direction = operator.index(direction)

        # Real transforms have only one valid direction. Validate it here
        # because the corresponding cuFFT execution routines ignore the flag.
        if self.fft_type in {_R2C, _D2Z}:
            if direction != CUFFT_FORWARD:
                raise ValueError(
                    "Real-to-complex transforms require CUFFT_FORWARD (-1)"
                )
        elif self.fft_type in {_C2R, _Z2D}:
            if direction != CUFFT_INVERSE:
                raise ValueError(
                    "Complex-to-real transforms require CUFFT_INVERSE (1)"
                )
        elif direction not in {CUFFT_FORWARD, CUFFT_INVERSE}:
            raise ValueError(
                "Direction must be CUFFT_FORWARD (-1) or CUFFT_INVERSE (1)"
                )
        if self.batch == 0:
            return
        if not self._work_area_bound:
            raise RuntimeError(
                "No work area is bound; call set_work_area() or "
                "allocate_work_area() first"
            )

        # The CuPy backend already owns the handle, workspace, and dispatch.
        if self.use_cupy:
            self._cupy_plan.fft(a, out, direction)
            return

        # Associate this execution with the caller's current stream. This is
        # also what serialises plans sharing one workspace in normal FLUCS use.
        nvcufft.set_stream(self.handle, int(cp.cuda.get_current_stream().ptr))

        # nvmath's low-level bindings operate on raw device pointers. Select
        # the execution routine matching the transform precision and direction.
        input_ptr = int(a.data.ptr)
        output_ptr = int(out.data.ptr)
        if self.fft_type == _C2C:
            nvcufft.exec_c2c(self.handle, input_ptr, output_ptr, direction)
        elif self.fft_type == _R2C:
            nvcufft.exec_r2c(self.handle, input_ptr, output_ptr)
        elif self.fft_type == _C2R:
            nvcufft.exec_c2r(self.handle, input_ptr, output_ptr)
        elif self.fft_type == _Z2Z:
            nvcufft.exec_z2z(self.handle, input_ptr, output_ptr, direction)
        elif self.fft_type == _D2Z:
            nvcufft.exec_d2z(self.handle, input_ptr, output_ptr)
        elif self.fft_type == _Z2D:
            nvcufft.exec_z2d(self.handle, input_ptr, output_ptr)
        else:  # guarded during construction; retained defensively
            raise RuntimeError(f"unsupported cuFFT type: {self.fft_type}")

    def close(self) -> None:
        """
        Destroy the cuFFT handle.

        The caller must finish queued FFTs before closing the plan.
        """

        if self.handle == 0:
            return

        # Destruction must occur with the plan's device active. CuPy plans are
        # released through their Python owner; custom handles are destroyed
        # directly through nvmath.
        handle = self.handle
        with cp.cuda.Device(self.device_id):
            if self.use_cupy:
                # CuPy owns and destroys the handle. Never destroy it twice.
                self._cupy_plan = None
            else:
                nvcufft.destroy(handle)

        # Drop every allocation reference after the handle can no longer use
        # the workspace.
        self.handle = 0
        self.work_area = None
        self.work_area_ptr = 0
        self.work_area_size = 0
        self._work_area_owner = None
        self._work_area_bound = False

    def _require_open(self) -> None:
        if self.handle == 0:
            raise RuntimeError("the cuFFT plan is closed")

    def _destroy_after_failed_init(self) -> None:
        if self.handle == 0:
            return
        try:
            with cp.cuda.Device(self.device_id):
                if self.use_cupy:
                    self._cupy_plan = None
                else:
                    nvcufft.destroy(self.handle)
        except BaseException:
            pass
        self.handle = 0

    def __copy__(self) -> FlucsPlanNd:
        raise TypeError("cuFFT plan handles cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> FlucsPlanNd:
        raise TypeError("cuFFT plan handles cannot be deep-copied")

    def __del__(self) -> None:
        # Interpreter shutdown can clear the imported CUDA modules before this
        # object is collected, so destruction must be best-effort here.
        try:
            self.close()
        except BaseException:
            pass

    def __repr__(self) -> str:
        state = "closed" if self.closed else f"handle={self.handle}"
        return (
            f"{type(self).__name__}(shape={self.shape}, "
            f"fft_type={_TYPE_NAMES[self.fft_type]}, batch={self.batch}, "
            f"use_cupy={self.use_cupy}, "
            f"work_size={self.work_size}, {state})"
        )


def allocate_shared_work_area(
    plans: Iterable[FlucsPlanNd | cp.cuda.cufft.PlanNd],
    allocator: Callable[[int], Any] | None = None,
    min_size: int = 0,
) -> Any | None:
    """
    Allocate one work area for the supplied custom-backend plans.

    Native CuPy plans and wrappers with use_cupy=True are silently skipped.
    Return None without allocating if no custom plans remain, including for an 
    empty iterable. Unrelated object types are rejected.

    Participating plans must belong to the same CUDA device. The default 
    allocator is cupy.cuda.Memory, which makes a direct CUDA device allocation. 
    Keep the returned object alive until all plans are closed or rebound.

    Allocates at least min_size bytes. Returns the memory object of the 
    allocator.

    """

    participating = []

    # Only custom plans can accept an external workspace. Native CuPy plans and
    # wrappers using the CuPy backend already own private allocations.
    for plan in plans:
        if isinstance(plan, FlucsPlanNd):
            if not plan.use_cupy:
                plan._require_open()
                participating.append(plan)
        elif not isinstance(plan, cp.cuda.cufft.PlanNd):
            raise TypeError(
                "All entries must be FlucsPlanNd or cupy.cuda.cufft.PlanNd "
                "instances"
                )
    plans = tuple(participating)
    if not plans:
        return None

    # One device allocation cannot be shared by plans on different GPUs.
    device_ids = {plan.device_id for plan in plans}
    if len(device_ids) != 1:
        raise ValueError("All plans sharing a work area must be on one device")
    device_id = device_ids.pop()
    required_size = max(plan.work_size for plan in plans)

    # Permit callers to reserve more than cuFFT currently requires, for example
    # when the allocation is also reused by another sequential operation.
    required_size = max(required_size, min_size)

    if required_size == 0:
        for plan in plans:
            plan._work_area_bound = True
        return None

    if allocator is None:
        allocator = cp.cuda.Memory

    # Allocate only the largest requirement because participating executions
    # are serialised, then bind the same allocation to every plan.
    with cp.cuda.Device(device_id):
        work_area = allocator(required_size)
        for plan in plans:
            plan.set_work_area(work_area)
    return work_area


def _positive_int(name: str, value: Any) -> int:
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int_tuple(name: str, values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(operator.index(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(value < 1 for value in result):
        raise ValueError(f"All entries of {name} must be positive")
    return result


def _optional_positive_int_tuple(
    name: str,
    values: Iterable[int] | None,
    rank: int,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    result = _positive_int_tuple(name, values)
    if len(result) != rank:
        raise ValueError(f"{name} must have the same length as shape")
    return result


def _workspace_pointer_and_size(
    work_area: Any | int,
    explicit_size: int | None,
) -> tuple[int, int, Any | None]:
    # A raw pointer has no Python object whose lifetime or size can be inferred.
    raw_pointer = isinstance(work_area, int)
    retained_object = None if raw_pointer else work_area

    if raw_pointer:
        ptr = int(work_area)
    elif hasattr(work_area, "ptr"):
        ptr = int(work_area.ptr)
    elif hasattr(work_area, "data") and hasattr(work_area.data, "ptr"):
        ptr = int(work_area.data.ptr)
    else:
        raise TypeError(
            "work_area must be an integer device pointer or an object exposing "
            ".ptr or .data.ptr"
        )

    # Prefer the object's accessible capacity, but permit an explicit smaller
    # region. Raw pointers always require an explicit byte count.
    inferred_size = _object_available_bytes(work_area, ptr)
    if explicit_size is None:
        if inferred_size is None:
            raise ValueError(
                "work_area_size is required because the allocation size cannot "
                "be inferred"
            )
        capacity = inferred_size
    else:
        capacity = int(explicit_size)
        if capacity < 0:
            raise ValueError("work_area_size must be non-negative")
        if inferred_size is not None and capacity > inferred_size:
            raise ValueError(
                f"work_area_size={capacity} exceeds the inferred accessible "
                f"size of {inferred_size} bytes"
            )

    return ptr, capacity, retained_object


def _object_available_bytes(obj: Any, ptr: int) -> int | None:
    if isinstance(obj, int):
        return None

    nbytes = getattr(obj, "nbytes", None)
    if nbytes is not None:
        # An array can be used as untyped workspace only when its bytes occupy
        # one contiguous region.
        flags = getattr(obj, "flags", None)
        if flags is not None:
            c_contiguous = bool(getattr(flags, "c_contiguous", False))
            f_contiguous = bool(getattr(flags, "f_contiguous", False))
            if not (c_contiguous or f_contiguous):
                raise ValueError("A CuPy-array work area must be contiguous")
        return int(nbytes)

    # A CuPy MemoryPointer exposes its allocation as .mem.  Account for a
    # pointer that starts partway through that allocation.
    memory = getattr(obj, "mem", None)
    if memory is not None and hasattr(memory, "ptr") and hasattr(memory, "size"):
        offset = ptr - int(memory.ptr)
        available = int(memory.size) - offset
        if offset < 0 or available < 0:
            raise ValueError("work-area pointer lies outside its allocation")
        return available

    size = getattr(obj, "size", None)
    if size is not None:
        # Bare CuPy Memory objects expose their capacity directly as size.
        return int(size)
    return None


def _object_device_id(obj: Any | None) -> int | None:
    if obj is None or isinstance(obj, int):
        return None

    device = getattr(obj, "device", None)
    if device is not None and hasattr(device, "id"):
        # CuPy arrays expose a Device object.
        return int(device.id)

    device_id = getattr(obj, "device_id", None)
    if device_id is not None:
        # CuPy Memory allocations expose the integer directly.
        return int(device_id)

    memory = getattr(obj, "mem", None)
    if memory is not None:
        # MemoryPointer objects expose the device through their base allocation.
        device_id = getattr(memory, "device_id", None)
        if device_id is not None:
            return int(device_id)
    return None


__all__ = [
    "CUFFT_FORWARD",
    "CUFFT_INVERSE",
    "FlucsPlanNd",
    "allocate_shared_work_area",
]
