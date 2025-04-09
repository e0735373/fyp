# https://nvidia.github.io/bionemo-framework/API_reference/bionemo/moco/interpolants/discrete_time/discrete/d3pm/

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Type, TypeVar
from abc import ABC, abstractmethod
from torch import Tensor
from enum import Enum

AnyEnum = TypeVar("AnyEnum", bound=Enum)

def string_to_enum(value: Union[str, AnyEnum], enum_type: Type[AnyEnum]) -> AnyEnum:
    """Converts a string to an enum value of the specified type. If the input is already an enum instance, it is returned as-is.

    Args:
        value (Union[str, E]): The string to convert or an existing enum instance.
        enum_type (Type[E]): The enum type to convert to.

    Returns:
        E: The corresponding enum value.

    Raises:
        ValueError: If the string does not correspond to any enum member.
    """
    if isinstance(value, enum_type):
        # If the value is already an enum, return it
        return value

    try:
        # Match the value to the Enum, case-insensitively
        return enum_type(value)
    except ValueError:
        # Raise a helpful error if the value is invalid
        valid_values = [e.value for e in enum_type]
        raise ValueError(f"Invalid value '{value}'. Expected one of {valid_values}.")


class PriorDistribution(ABC):
    """An abstract base class representing a prior distribution."""

    @abstractmethod
    def sample(self, shape: Tuple, mask: Optional[Tensor] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generates a specified number of samples from the time distribution.

        Args:
        shape (Tuple): The shape of the samples to generate.
        mask (Optional[Tensor], optional): A tensor indicating which samples should be masked. Defaults to None.
        device (str, optional): The device on which to generate the samples. Defaults to "cpu".

        Returns:
            float: A tensor of samples.
        """
        pass


class DiscretePriorDistribution(PriorDistribution):
    """An abstract base class representing a discrete prior distribution."""

    def __init__(self, num_classes: int, prior_dist: Tensor):
        """Initializes a DiscretePriorDistribution instance.

        Args:
        num_classes (int): The number of classes in the discrete distribution.
        prior_dist (Tensor): The prior distribution over the classes.

        Returns:
        None
        """
        self.num_classes = num_classes
        self.prior_dist = prior_dist

    def get_num_classes(self) -> int:
        """Getter for num_classes."""
        return self.num_classes

    def get_prior_dist(self) -> Tensor:
        """Getter for prior_dist."""
        return self.prior_dist

class DiscreteUniformPrior(DiscretePriorDistribution):
    """A subclass representing a discrete uniform prior distribution."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initializes a discrete uniform prior distribution.

        Args:
            num_classes (int): The number of classes in the discrete uniform distribution. Defaults to 10.
        """
        prior_dist = torch.ones((num_classes)) * 1 / num_classes
        super().__init__(num_classes, prior_dist)
        if torch.sum(self.prior_dist).item() - 1.0 > 1e-5:
            raise ValueError("Prior distribution probabilities do not sum up to 1.0")

    def sample(
        self,
        shape: Tuple,
        mask: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Generates a specified number of samples.

        Args:
            shape (Tuple): The shape of the samples to generate.
            device (str): cpu or gpu.
            mask (Optional[Tensor]): An optional mask to apply to the samples. Defaults to None.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            float: A tensor of samples.
        """
        samples = torch.randint(0, self.num_classes, shape, device=device, generator=rng_generator)
        if mask is not None:
            samples = samples * mask[(...,) + (None,) * (len(samples.shape) - len(mask.shape))]
        return samples

class DiscreteMaskedPrior(DiscretePriorDistribution):
    """A subclass representing a Discrete Masked prior distribution."""

    def __init__(self, num_classes: int = 10, mask_dim: Optional[int] = None, inclusive: bool = True) -> None:
        """Discrete Masked prior distribution.

        Theres 3 ways I can think of defining the problem that are hard to mesh together.

        1. [..., M, ....] inclusive anywhere --> exisiting LLM tokenizer where the mask has a specific location not at the end
        2. [......, M] inclusive on end --> mask_dim = None with inclusive set to True default stick on the end
        3. [.....] + [M] exclusive --> the number of classes representes the number of data classes and one wishes to add a separate MASK dimension.
            - Note the pad_sample function is provided to help add this extra external dimension.

        Args:
            num_classes (int): The number of classes in the distribution. Defaults to 10.
            mask_dim (int): The index for the mask token. Defaults to num_classes - 1 if inclusive or num_classes if exclusive.
            inclusive (bool): Whether the mask is included in the specified number of classes.
                                If True, the mask is considered as one of the classes.
                                If False, the mask is considered as an additional class. Defaults to True.
        """
        if inclusive:
            if mask_dim is None:
                mask_dim = num_classes - 1
            else:
                if mask_dim >= num_classes:
                    raise ValueError(
                        "As Inclusive accounts for the mask as one of the specified num_classes, the provided mask_dim cannot be >= to num_classes"
                    )
            prior_dist = torch.zeros((num_classes))
            prior_dist[-1] = 1.0
            super().__init__(num_classes, prior_dist)
            self.mask_dim = mask_dim
        else:
            prior_dist = torch.zeros((num_classes + 1))
            prior_dist[-1] = 1.0
            super().__init__(num_classes + 1, prior_dist)
            self.mask_dim = num_classes
        if torch.sum(self.prior_dist).item() - 1.0 >= 1e-5:
            raise ValueError("Invalid probability distribution. Must sum to 1.0")

    def sample(
        self,
        shape: Tuple,
        mask: Optional[Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Generates a specified number of samples.

        Args:
            shape (Tuple): The shape of the samples to generate.
            device (str): cpu or gpu.
            mask (Optional[Tensor]): An optional mask to apply to the samples. Defaults to None.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            Float: A tensor of samples.
        """
        samples = torch.ones(shape, dtype=torch.int64, device=device) * self.mask_dim
        if mask is not None:
            samples = samples * mask[(...,) + (None,) * (len(samples.shape) - len(mask.shape))]
        return samples

    def is_masked(self, sample: Tensor) -> Tensor:
        """Creates a mask for whether a state is masked.

        Args:
            sample (Tensor): The sample to check.

        Returns:
            Tensor: A float tensor indicating whether the sample is masked.
        """
        return (sample == self.mask_dim).float()

    def pad_sample(self, sample: Tensor) -> Tensor:
        """Pads the input sample with zeros along the last dimension.

        Args:
            sample (Tensor): The input sample to be padded.

        Returns:
            Tensor: The padded sample.
        """
        # Create a zeros tensor with the same shape as the original tensor, except the last dimension is 1
        zeros = torch.zeros((*sample.shape[:-1], 1), dtype=torch.float, device=sample.device)
        # Concatenate along the last dimension to make the shape (..., N+1)
        padded_sample = torch.cat((sample, zeros), dim=-1)
        return padded_sample

class TimeDistribution(ABC):
    """An abstract base class representing a time distribution.

    Args:
        discrete_time (bool): Whether the time is discrete.
        nsteps (Optional[int]): Number of nsteps for discretization.
        min_t (Optional[float]): Min continuous time.
        max_t (Optional[float]): Max continuous time.
        rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
    """

    def __init__(
        self,
        discrete_time: bool = False,
        nsteps: Optional[int] = None,
        min_t: Optional[float] = None,
        max_t: Optional[float] = None,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes a TimeDistribution object."""
        self.discrete_time = discrete_time
        self.nsteps = nsteps
        self.rng_generator = rng_generator
        if discrete_time:
            min_t = 0.0
            max_t = 1.0
            if nsteps is None:
                raise ValueError("nsteps must not be None and must be specified for discrete time")
        if min_t is not None and isinstance(min_t, float):
            if not 0 <= min_t < 1.0:
                raise ValueError("min_t must be greater than or equal to 0 and less than 1.0")
        self.min_t = min_t
        if max_t is not None and isinstance(max_t, float):
            if not 0 < max_t <= 1.0:
                raise ValueError("max_t must be greater than 0 and less than or equal to 1.0")
        self.max_t = max_t
        if (
            self.min_t is not None
            and self.max_t is not None
            and isinstance(self.min_t, float)
            and isinstance(self.max_t, float)
        ):
            if self.min_t >= self.max_t:
                raise ValueError("min_t must be less than max_t")

    @abstractmethod
    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ) -> float:
        """Generates a specified number of samples from the time distribution.

        Args:
        n_samples (int): The number of samples to generate.
        device (str): cpu or gpu.
        rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            float: A list or array of samples.
        """
        pass

class UniformTimeDistribution(TimeDistribution):
    """A class representing a uniform time distribution."""

    def __init__(
        self,
        min_t: float = 0.0,
        max_t: float = 1.0,
        discrete_time: bool = False,
        nsteps: Optional[int] = None,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes a UniformTimeDistribution object.

        Args:
            min_t (float): The minimum time value.
            max_t (float): The maximum time value.
            discrete_time (Bool): Whether the time is discrete.
            nsteps (Optional[int]): Number of nsteps for discretization.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        super().__init__(discrete_time, nsteps, min_t, max_t, rng_generator)

    def sample(
        self,
        n_samples: Union[int, Tuple[int, ...], torch.Size],
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Generates a specified number of samples from the uniform time distribution.

        Args:
            n_samples (int): The number of samples to generate.
            device (str): cpu or gpu.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.

        Returns:
            A tensor of samples.
        """
        if rng_generator is None:
            rng_generator = self.rng_generator
        if self.discrete_time:
            if self.nsteps is None:
                raise ValueError("nsteps cannot be None for discrete time sampling")
            time_step = torch.randint(
                0,
                self.nsteps,
                size=(n_samples,) if isinstance(n_samples, int) else n_samples,
                device=device,
                generator=rng_generator,
            )
        else:
            time_step = torch.rand(n_samples, device=device, generator=rng_generator)
            if self.min_t and self.max_t and self.min_t > 0:
                time_step = time_step * (self.max_t - self.min_t) + self.min_t
        return time_step

class TimeDirection(Enum):
    """Enum for the direction of the noise schedule."""

    UNIFIED = "unified"  # Noise(0) --> Data(1)
    DIFFUSION = "diffusion"  # Noise(1) --> Data(0)

class DiscreteNoiseSchedule(ABC):
    """A base class for discrete noise schedules."""

    def __init__(self, nsteps: int, direction: TimeDirection):
        """Initialize the DiscreteNoiseSchedule.

        Args:
           nsteps (int): number of discrete steps.
           direction (TimeDirection): required this defines in which direction the scheduler was built
        """
        self.nsteps = nsteps
        self.direction = string_to_enum(direction, TimeDirection)

    def generate_schedule(
        self,
        nsteps: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None,
    ) -> Tensor:
        """Generate the noise schedule as a tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
            synchronize (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
                this parameter allows to flip the direction to match the specified one (default is None).
        """
        schedule = self._generate_schedule(nsteps, device)
        if synchronize and self.direction != string_to_enum(synchronize, TimeDirection):
            return torch.flip(schedule, dims=[0])
        else:
            return schedule

    @abstractmethod
    def _generate_schedule(self, nsteps: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generate the noise schedule tensor.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        pass

    def calculate_derivative(
        self,
        nsteps: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        synchronize: Optional[TimeDirection] = None,
    ) -> Tensor:
        """Calculate the time derivative of the schedule.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
            synchronize (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction,
                this parameter allows to flip the direction to match the specified one (default is None).

        Returns:
            Tensor: A tensor representing the time derivative of the schedule.

        Raises:
            NotImplementedError: If the derivative calculation is not implemented for this schedule.
        """
        raise NotImplementedError("Derivative calculation is not implemented for this schedule.")


class DiscreteCosineNoiseSchedule(DiscreteNoiseSchedule):
    """A cosine discrete noise schedule."""

    def __init__(self, nsteps: int, nu: float = 1.0, s: float = 0.008):
        """Initialize the CosineNoiseSchedule.

        Args:
            nsteps (int): Number of discrete steps.
            nu (Optional[float]): Hyperparameter for the cosine schedule exponent (default is 1.0).
            s (Optional[float]): Hyperparameter for the cosine schedule shift (default is 0.008).
        """
        super().__init__(nsteps=nsteps, direction=TimeDirection.DIFFUSION)
        self.nu = nu
        self.s = s

    def _generate_schedule(self, nsteps: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
        """Generate the cosine noise schedule.

        Args:
            nsteps (Optional[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        if nsteps is None:
            nsteps = self.nsteps
        steps = (
            nsteps + 1
        )  #! matches OpenAI code https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L62
        x = torch.linspace(0, nsteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / nsteps) ** self.nu + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.001, 0.999)
        return 1 - betas

    def _clip_noise_schedule(self, alphas2: Tensor, clip_value: float = 0.001) -> Tensor:
        """For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during sampling.

        Args:
            alphas2 (Tensor): The noise schedule given by alpha^2.
            clip_value (Optional[float]): The minimum value for alpha_t / alpha_t-1 (default is 0.001).

        Returns:
            Tensor: The clipped noise schedule.
        """
        alphas2 = torch.cat([torch.ones(1, device=alphas2.device), alphas2], dim=0)

        alphas_step = alphas2[1:] / alphas2[:-1]

        alphas_step = torch.clamp(alphas_step, min=clip_value, max=1.0)
        alphas2 = torch.cumprod(alphas_step, dim=0)

        return alphas2


class InferenceSchedule(ABC):
    """A base class for inference time schedules."""

    def __init__(
        self,
        nsteps: int,
        min_t: float = 0,
        padding: float = 0,
        dilation: float = 0,
        direction: Union[TimeDirection, str] = TimeDirection.UNIFIED,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the InferenceSchedule.

        Args:
            nsteps (int): Number of time steps.
            min_t (float): minimum time value defaults to 0.
            padding (float): padding time value defaults to 0.
            dilation (float): dilation time value defaults to 0 ie the number of replicates.
            direction (Optional[str]): TimeDirection to synchronize the schedule with. If the schedule is defined with a different direction, this parameter allows to flip the direction to match the specified one (default is None).
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        """
        self.nsteps = nsteps
        self.min_t = min_t
        self.padding = padding
        self.dilation = dilation
        self.direction = string_to_enum(direction, TimeDirection)
        self.device = device

    @abstractmethod
    def generate_schedule(
        self, nsteps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        """Generate the time schedule as a tensor.

        Args:
            nsteps (Optioanl[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").
        """
        pass

    def pad_time(
        self, n_samples: int, scalar_time: float, device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        """Creates a tensor of shape (n_samples,) filled with a scalar time value.

        Args:
            n_samples (int): The desired dimension of the output tensor.
            scalar_time (float): The scalar time value to fill the tensor with.
            device (Optional[Union[str, torch.device]], optional):
                The device to place the tensor on. Defaults to None, which uses the default device.

        Returns:
            Tensor: A tensor of shape (n_samples,) filled with the scalar time value.
        """
        return torch.full((n_samples,), fill_value=scalar_time).to(device)

class DiscreteInferenceSchedule(InferenceSchedule):
    """A base class for discrete time inference schedules."""

    def discretize(
        self,
        nsteps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Discretize the time schedule into a list of time deltas.

        Args:
            nsteps (Optioanl[int]): Number of time steps. If None, uses the value from initialization.
            device (Optional[str]): Device to place the schedule on (default is "cpu").

        Returns:
            Tensor: A tensor of time deltas.
        """
        if self.padding > 0 or self.dilation > 0:
            raise NotImplementedError("discreteize is not implemented for discrete schedules with padding or dilation")
        if device is None:
            device = self.device
        return torch.full(
            (nsteps if nsteps is not None else self.nsteps,),
            1 / (nsteps if nsteps is not None else self.nsteps),
            device=device,
        )


def safe_index(tensor: Tensor, index: Tensor, device: Optional[torch.device]):
    """Safely indexes a tensor using a given index and returns the result on a specified device.

    Note can implement forcing with  return tensor[index.to(tensor.device)].to(device) but has costly migration.

    Args:
        tensor (Tensor): The tensor to be indexed.
        index (Tensor): The index to use for indexing the tensor.
        device (torch.device): The device on which the result should be returned.

    Returns:
        Tensor: The indexed tensor on the specified device.

    Raises:
        ValueError: If tensor, index are not all on the same device.
    """
    if not (tensor.device == index.device):
        raise ValueError(
            f"Tensor, index, and device must all be on the same device. "
            f"Got tensor.device={tensor.device}, index.device={index.device}, and device={device}."
        )

    return tensor[index].to(device)

class Interpolant(ABC):
    """An abstract base class representing an Interpolant.

    This class serves as a foundation for creating interpolants that can be used
    in various applications, providing a basic structure and interface for
    interpolation-related operations.
    """

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: PriorDistribution,
        device: Union[str, torch.device] = "cpu",
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes the Interpolant class.

        Args:
            time_distribution (TimeDistribution): The distribution of time steps.
            prior_distribution (PriorDistribution): The prior distribution of the variable.
            device (Union[str, torch.device], optional): The device on which to operate. Defaults to "cpu".
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        self.time_distribution = time_distribution
        self.prior_distribution = prior_distribution
        self.device = device
        self.rng_generator = rng_generator

    @abstractmethod
    def interpolate(self, *args, **kwargs) -> Tensor:
        """Get x(t) with given time t from noise and data.

        Interpolate between x0 and x1 at the given time t.
        """
        pass

    @abstractmethod
    def step(self, *args, **kwargs) -> Tensor:
        """Do one step integration."""
        pass

    def general_step(self, method_name: str, kwargs: dict):
        """Calls a step method of the class by its name, passing the provided keyword arguments.

        Args:
            method_name (str): The name of the step method to call.
            kwargs (dict): Keyword arguments to pass to the step method.

        Returns:
            The result of the step method call.

        Raises:
            ValueError: If the provided method name does not start with 'step'.
            Exception: If the step method call fails. The error message includes a list of available step methods.

        Note:
            This method allows for dynamic invocation of step methods, providing flexibility in the class's usage.
        """
        if not method_name.startswith("step"):
            raise ValueError(f"Method name '{method_name}' does not start with 'step'")

        try:
            # Get the step method by its name
            func = getattr(self, method_name)
            # Call the step method with the provided keyword arguments
            return func(**kwargs)
        except Exception as e:
            # Get a list of available step methods
            available_methods = "\n".join([f"  - {attr}" for attr in dir(self) if attr.startswith("step")])
            # Create a detailed error message
            error_message = f"Error calling method '{method_name}': {e}\nAvailable step methods:\n{available_methods}"
            # Re-raise the exception with the detailed error message
            raise type(e)(error_message)

    def sample_prior(self, *args, **kwargs) -> Tensor:
        """Sample from prior distribution.

        This method generates a sample from the prior distribution specified by the
        `prior_distribution` attribute.

        Returns:
            Tensor: The generated sample from the prior distribution.
        """
        # Ensure the device is specified, default to self.device if not provided
        if "device" not in kwargs:
            kwargs["device"] = self.device
        kwargs["rng_generator"] = self.rng_generator
        # Sample from the prior distribution
        return self.prior_distribution.sample(*args, **kwargs)

    def sample_time(self, *args, **kwargs) -> Tensor:
        """Sample from time distribution."""
        # Ensure the device is specified, default to self.device if not provided
        if "device" not in kwargs:
            kwargs["device"] = self.device
        kwargs["rng_generator"] = self.rng_generator
        # Sample from the time distribution
        return self.time_distribution.sample(*args, **kwargs)

    def to_device(self, device: str):
        """Moves all internal tensors to the specified device and updates the `self.device` attribute.

        Args:
            device (str): The device to move the tensors to (e.g. "cpu", "cuda:0").

        Note:
            This method is used to transfer the internal state of the DDPM interpolant to a different device.
            It updates the `self.device` attribute to reflect the new device and moves all internal tensors to the specified device.
        """
        self.device = device
        for attr_name in dir(self):
            if attr_name.startswith("_") and isinstance(getattr(self, attr_name), torch.Tensor):
                setattr(self, attr_name, getattr(self, attr_name).to(device))
        return self

    def clean_mask_center(self, data: Tensor, mask: Optional[Tensor] = None, center: bool = False) -> Tensor:
        """Returns a clean tensor that has been masked and/or centered based on the function arguments.

        Args:
            data: The input data with shape (..., nodes, features).
            mask: An optional mask to apply to the data with shape (..., nodes). If provided, it is used to calculate the CoM. Defaults to None.
            center: A boolean indicating whether to center the data around the calculated CoM. Defaults to False.

        Returns:
            The data with shape (..., nodes, features) either centered around the CoM if `center` is True or unchanged if `center` is False.
        """
        if mask is not None:
            data = data * mask.unsqueeze(-1)
        if not center:
            return data
        if mask is None:
            num_nodes = torch.tensor(data.shape[1], device=data.device)
        else:
            num_nodes = torch.clamp(mask.sum(dim=-1), min=1)  # clamp used to prevent divide by 0
        com = data.sum(dim=-2) / num_nodes.unsqueeze(-1)
        return data - com.unsqueeze(-2)


def _is_one_hot(data, num_classes):
    """Check if data is one-hot encoded.

    Parameters:
    - data (Tensor): Input data to check.
    - num_classes (int): Expected number of classes for one-hot encoding.

    Returns:
    - bool: True if data is one-hot encoded, False otherwise.
    """
    if len(data.shape) < 2 or data.shape[-1] != num_classes:
        return False  # Not one-hot if last dim doesn't match num_classes or less than 2D

    # Check if all vectors are one-hot
    return (data.sum(dim=-1) == 1).all() and (data.flatten().shape[0] / num_classes) % 1 == 0


class D3PM(Interpolant):
    """A Discrete Denoising Diffusion Probabilistic Model (D3PM) interpolant."""

    def __init__(
        self,
        time_distribution: TimeDistribution,
        prior_distribution: DiscretePriorDistribution,
        noise_schedule: DiscreteNoiseSchedule,
        device: str = "cpu",
        last_time_idx: int = 0,
        rng_generator: Optional[torch.Generator] = None,
    ):
        """Initializes the D3PM interpolant.

        Args:
            time_distribution (TimeDistribution): The distribution of time steps, used to sample time points for the diffusion process.
            prior_distribution (PriorDistribution): The prior distribution of the variable, used as the starting point for the diffusion process.
            noise_schedule (DiscreteNoiseSchedule): The schedule of noise, defining the amount of noise added at each time step.
            device (str, optional): The device on which to run the interpolant, either "cpu" or a CUDA device (e.g. "cuda:0"). Defaults to "cpu".
            last_time_idx (int, optional): The last time index to consider in the interpolation process. Defaults to 0.
            rng_generator: An optional :class:`torch.Generator` for reproducible sampling. Defaults to None.
        """
        # We initialize with CPU due to numerical precision issues on A100 that are not observed on A6000
        super().__init__(time_distribution, prior_distribution, "cpu", rng_generator)
        self.noise_schedule = noise_schedule
        self._loss_function = nn.CrossEntropyLoss(reduction="none")
        self.timesteps = noise_schedule.nsteps
        self.num_classes = prior_distribution.num_classes
        self.terminal_distribution = prior_distribution.prior_dist.to(self.device)
        self._initialize_schedules(self.device)
        self.last_time_idx = last_time_idx
        self.to_device(device)

    def _get_Qt(self, alphas: Tensor) -> Tensor:
        """Calculate the transition matrix Qt based on the terminal distribution.

        The transition matrix Qt represents the probabilities of transitioning from one state to another at a given time step.
        It is calculated based on the terminal distribution, which can be either uniform, a mask, or a custom distribution.
        See Appendix A.2 D3PM https://arxiv.org/pdf/2107.03006 which shows what happens for various prior distributions.

        The terminal distribution can be:
        - Uniform: a uniform distribution over all states.
        - Mask: a mask where the last dimension is 1 and the rest are 0.
        - Custom: a custom distribution provided by the user.

        Args:
            alphas (Tensor): A tensor of probabilities, where each alpha represents the probability of staying in a state at a given time step.

        Returns:
            Tensor: The transition matrix Qt.
        """
        QT = []
        for alpha_t in alphas:
            stay_prob = torch.eye(len(self.terminal_distribution), device=self.device) * alpha_t
            diffuse_prob = (1.0 - alpha_t) * (
                torch.ones(1, len(self.terminal_distribution), device=self.device)
                * (self.terminal_distribution.unsqueeze(0))
            )
            QT.append(stay_prob + diffuse_prob)
        return torch.stack(QT, dim=0)

    def _calculate_transition_matrix(self, alphas: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the rate transition matrix `Qt`, its cumulative variant `Qt_bar`, and the cumulative variant of the previous time step `Qt_bar_prev`.

        Args:
            alphas (Tensor): A tensor of probabilities, where each alpha represents the probability of staying in a state at a given time step.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the rate transition matrix `Qt`, its cumulative variant `Qt_bar`, and the cumulative variant of the previous time step `Qt_bar_prev`.
        """
        Qt = self._get_Qt(alphas)
        Qt_prev = torch.eye(self.num_classes, device=self.device)
        Qt_bar = []
        for i in range(len(alphas)):
            Qtb = Qt_prev @ Qt[i]
            if torch.any((Qtb.sum(-1) - 1.0).abs() > 1e-4):
                raise ValueError(f"Invalid Distribution for Qt_bar at step {i}")
            Qt_bar.append(Qtb)
            Qt_prev = Qtb
        Qt_bar = torch.stack(Qt_bar)
        Qt_bar_prev = Qt_bar[:-1]
        Qt_prev_pad = torch.eye(self.num_classes, device=self.device)
        Qt_bar_prev = torch.concat([Qt_prev_pad.unsqueeze(0), Qt_bar_prev], dim=0)
        return Qt, Qt_bar, Qt_bar_prev

    def _initialize_schedules(self, device):
        """Initializes the transition matrices for the discrete diffusion process.

        This method computes the rate transition matrix `Qt` and its cumulative variants `Qt_bar` and `Qt_prev_bar`
        based on the provided noise schedule.

        Note:
            `Qt` represents the rate transition matrix, where `Qt[t]` is the transition matrix at time step `t`.
            `Qt_bar` and `Qt_prev_bar` are the cumulative variants of `Qt`, where `Qt_bar[t]` represents the cumulative
            transition matrix from time step `0` to `t`, and `Qt_prev_bar[t]` represents the cumulative transition matrix
            from time step `0` to `t-1`.

        Args:
            device (str): The device on which to compute the transition matrices.
        """
        if self.noise_schedule is None:
            raise ValueError("noise_schedule cannot be None for D3PM")
        alphas = self.noise_schedule.generate_schedule(device=device)
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self._alpha_bar = torch.exp(log_alpha_bar)
        #! Note to users that the tranditional cosine schedule is a very quick convergence of alpha. Pay close attention to the scheduler here
        Qt, Qt_bar, Qt_prev_bar = self._calculate_transition_matrix(alphas)
        self._Qt = Qt[-self.timesteps :]
        self._Qt_transposed = self._Qt.transpose(1, 2)
        self._Qt_bar = Qt_bar[-self.timesteps :]
        self._Qt_prev_bar = Qt_prev_bar[-self.timesteps :]

    def interpolate(self, data: Tensor, t: Tensor):
        """Interpolate using discrete interpolation method.

        This method implements Equation 2 from the D3PM paper (https://arxiv.org/pdf/2107.03006), which
        calculates the interpolated discrete state `xt` at time `t` given the input data and noise
        via q(xt|x0) = Cat(xt; p = x0*Qt_bar).

        Args:
            data (Tensor): The input data to be interpolated.
            t (Tensor): The time step at which to interpolate.

        Returns:
            Tensor: The interpolated discrete state `xt` at time `t`.
        """
        if not _is_one_hot(data, self.num_classes):
            x1_hot = F.one_hot(data, self.num_classes)
        else:
            x1_hot = data
        ford = safe_index(self._Qt_bar, t - self.last_time_idx, data.device)
        if x1_hot.ndim > 3:  # einsum precision issues on A100 not A6000 for 2D inputs
            ford_prep = ford
            for _ in range(x1_hot.ndim - 2):
                ford_prep = ford_prep.unsqueeze(1)
            probs = (x1_hot.float().unsqueeze(-2) * ford_prep).sum(dim=(-2))
        else:
            probs = torch.einsum("b...j, bji -> b...i", [x1_hot.float(), ford])
        if torch.any((probs.sum(-1) - 1.0).abs() > 1e-4):
            raise ValueError(
                f"**INVALID BEHAVIOR** Probability Distribution does not sum to 1.0 for time {t}. "
                f"**INVESTIGATE YOUR DEVICE PRECISION**: This error has been triggered before on A100 by initializing the Qt terms on gpu. "
                f"Normalized to ensure validity. Original sums: {probs.sum(-1)}",
            )
        xt = self._sample_categorical(torch.log(probs) + 1.0e-6)
        return xt

    def forward_process(self, data: Tensor, t: Tensor) -> Tensor:
        """Apply the forward process to the data at time t.

        Args:
            data (Tensor): target discrete ids
            t (Tensor): time

        Returns:
            Tensor: x(t) after applying the forward process
        """
        return self.interpolate(data, t)

    def _sample_categorical(self, logits, mask: Optional[Tensor] = None, temperature: float = 1.0) -> Tensor:
        """Sample a categorical distribution using the Gumbel-Softmax trick.

        This method samples a categorical distribution from the given logits,
        optionally applying a mask and using a specified temperature.

        Args:
            logits (Tensor): The logits of the categorical distribution.
            mask (Optional[Tensor], optional): An optional mask to apply to the noise added to logits. Defaults to None.
            temperature (float, optional): The temperature to use for the Gumbel-Softmax trick. Defaults to 1.0.

        Returns:
            Tensor: A sample from the categorical distribution.
        """
        # return torch.distributions.Categorical(logits=logits).sample()
        noise = torch.rand_like(logits)
        noise = torch.clip(noise, 1.0e-6, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        if mask is not None:
            sample = torch.argmax((logits / temperature) + gumbel_noise * mask, dim=-1)
        else:
            sample = torch.argmax((logits / temperature) + gumbel_noise, dim=-1)
        return sample

    def _q_posterior_logits(
        self, model_out: Tensor, t: Tensor, xt: Tensor, model_out_is_logits: bool = True
    ) -> Tensor:
        """Calculate the q-posterior logits using the predicted x0 and the current state xt at time t.

        This method implements Equation 3 from the D3PM paper (https://arxiv.org/pdf/2107.03006), which calculates the q-posterior
        distribution over the previous state x0 given the current state xt and the model output.

        Args:
            model_out (Tensor): The output of the model at the current time step.
            t (Tensor): The current time step.
            xt (Tensor): The current discrete state at time t.
            model_out_is_logits (bool, optional): A flag indicating whether the model output is already in logits form. If True, the output is assumed to be logits; otherwise, it is converted to logits. Defaults to True.

        Returns:
            Tensor: The q-posterior logits.
        """
        if not model_out_is_logits:  # model_out.dtype == torch.int64 or model_out.dtype == torch.int32:
            # Convert model output to logits if it's a categorical distribution
            x0_logits = torch.log(torch.nn.functional.one_hot(model_out, self.num_classes).float() + 1.0e-6)
        else:
            # Otherwise, assume model output is already logits
            x0_logits = model_out.clone()

        # Calculate xt_guess: the predicted probability of xt given x0 and t
        xt_guess = torch.einsum(
            "b...j, bji -> b...i",
            [
                torch.nn.functional.one_hot(xt, self.num_classes).float(),
                safe_index(self._Qt_transposed, t - self.last_time_idx, model_out.device),
            ],
        )

        # Calculate softmaxed x0_logits
        softmaxed = torch.softmax(x0_logits, dim=-1)  # bs, ..., num_classes

        # Calculate x0_guess: the predicted probability of x0 given xt and t-1
        x0_guess = torch.einsum(
            "b...c,bcd->b...d",
            softmaxed,
            safe_index(self._Qt_prev_bar, t - self.last_time_idx, model_out.device),
        )

        # Calculate q-posterior logits
        out = torch.log(xt_guess + 1.0e-6) + torch.log(x0_guess + 1.0e-6)
        t_broadcast = t.reshape((t.shape[0], *[1] * (xt.dim())))
        q_posterior_logits = torch.where(t_broadcast == self.last_time_idx, x0_logits, out)
        return q_posterior_logits

    def step(
        self,
        model_out: Tensor,
        t: Tensor,
        xt: Tensor,
        mask: Optional[Tensor] = None,
        temperature: float = 1.0,
        model_out_is_logits: bool = True,
    ):
        """Perform a single step in the discrete interpolant method, transitioning from the current discrete state `xt` at time `t` to the next state.

        This step involves:

        1. Computing the predicted q-posterior logits using the model output `model_out` and the current state `xt` at time `t`.
        2. Sampling the next state from the predicted q-posterior distribution using the Gumbel-Softmax trick.

        Args:
            model_out (Tensor): The output of the model at the current time step, which is used to compute the predicted q-posterior logits.
            t (Tensor): The current time step, which is used to index into the transition matrices and compute the predicted q-posterior logits.
            xt (Tensor): The current discrete state at time `t`, which is used to compute the predicted q-posterior logits and sample the next state.
            mask (Optional[Tensor], optional): An optional mask to apply to the next state, which can be used to mask out certain tokens or regions. Defaults to None.
            temperature (float, optional): The temperature to use for the Gumbel-Softmax trick, which controls the randomness of the sampling process. Defaults to 1.0.
            model_out_is_logits (bool, optional): A flag indicating whether the model output is already in logits form. If True, the output is assumed to be logits; otherwise, it is converted to logits. Defaults to True.

        Returns:
            Tensor: The next discrete state at time `t-1`.
        """
        pred_q_posterior_logits = self._q_posterior_logits(model_out, t, xt, model_out_is_logits)
        nonzero_mask = (t != self.last_time_idx).to(xt.dtype).reshape(xt.shape[0], *([1] * (len(xt.shape))))
        x_next = self._sample_categorical(pred_q_posterior_logits, nonzero_mask, temperature=temperature)
        # # Apply mask if provided
        if mask is not None:
            x_next = x_next * mask
        return x_next

    def loss(
        self,
        logits: Tensor,
        target: Tensor,
        xt: Tensor,
        time: Tensor,
        mask: Optional[Tensor] = None,
        loss_reweight: bool = False,
        vb_scale: float = 0.0,
    ):
        """Calculate the cross-entropy loss between the model prediction and the target output.

        The loss is calculated between the batch x node x class logits and the target batch x node. If a mask is provided, the loss is
        calculated only for the non-masked elements. Additionally, if vb_scale is greater than 0, the variational lower bound loss is
        calculated and added to the total loss.

        Args:
            logits (Tensor): The predicted output from the model, with shape batch x node x class.
            target (Tensor): The target output for the model prediction, with shape batch x node.
            xt (Tensor): The current data point.
            time (Tensor): The time at which the loss is calculated.
            mask (Optional[Tensor], optional): The mask for the data point. Defaults to None.
            vb_scale (float, optional): The scale factor for the variational lower bound loss. Defaults to 0.0.

        Returns:
            Tensor: The calculated loss tensor. If aggregate is True, the loss and variational lower bound loss are aggregated and
            returned as a single tensor. Otherwise, the loss and variational lower bound loss are returned as separate tensors.
        """
        assert target.ndim + 1 == logits.ndim
        loss = self._loss_function(logits.transpose(-1, 1), target.long())

        if mask is not None:
            loss = loss * mask
            num_non_masked_elements = torch.sum(mask, dim=-1)
        else:
            num_non_masked_elements = logits.size(1)

        if loss_reweight:
            loss_prefix_sum = torch.cumsum(loss, dim=-1) - loss

            # alpha * e^{-pl_i / beta} * l_i
            # pl_i = \sum_{j<i} l_j
            # beta = num_non_masked_elements
            # alpha = 1

            loss_multiplier = torch.exp(-loss_prefix_sum / num_non_masked_elements.unsqueeze(-1))

            loss = loss_multiplier * loss

        loss = torch.sum(loss, dim=(-1)) / num_non_masked_elements

        if vb_scale > 0:
            target = F.one_hot(target, num_classes=self.num_classes).float()
            true_q_posterior_logits = self._q_posterior_logits(target, time, xt)
            pred_q_posterior_logits = self._q_posterior_logits(logits, time, xt)
            vb_loss = self._variational_lower_bound(true_q_posterior_logits, pred_q_posterior_logits)
            vb_loss = vb_scale * vb_loss
        else:
            vb_loss = 0
        if vb_scale > 0:
            loss += vb_loss
        return loss

    def _variational_lower_bound(self, dist1: Tensor, dist2: Tensor) -> Tensor:
        """Calculate the variational lower bound (VLB) between two distributions.

        The VLB measures the difference between the true and approximate posterior distributions.
        It is used to regularize the model and encourage it to produce more accurate predictions.

        Args:
            dist1 (Tensor): The true posterior distribution.
            dist2 (Tensor): The approximate posterior distribution.

        Returns:
            Tensor: The variational lower bound loss.
        """
        # Flatten dist1 and dist2 to simplify calculations
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        # Calculate the VLB
        out = torch.softmax(dist1 + 1.0e-6, dim=-1) * (
            torch.log_softmax(dist1 + 1.0e-6, dim=-1) - torch.log_softmax(dist2 + 1.0e-6, dim=-1)
        )
        # Return the mean of the VLB across all elements
        return out.sum(dim=-1).mean()
