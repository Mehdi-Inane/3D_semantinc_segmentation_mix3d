import math
import random
from warnings import warn
from copy import deepcopy
from . import functional as F
from .composition import add_metaclass,format_args,SerializableMeta


__all__ = [
    "Scale3d",
    "RotateAroundAxis3d",
    "Crop3d",
    "RandomMove3d",
    "Move3d",
    "Center3d",
    "RandomDropout3d",
    "Flip3d",
]


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple

    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)

@add_metaclass(SerializableMeta)
class BasicTransform:
    call_backup = None

    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params = {}
        self.replay_mode = False
        self.applied_in_replay = False

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(
                    key in kwargs for key in self.targets_as_params
                ), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(
                    targets_as_params
                )
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname()
                        + " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(
        self, params, force_apply=False, **kwargs
    ):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {
                    k: kwargs[k] for k in self.target_dependence.get(key, [])
                }
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def set_deterministic(self, flag, save_key="replay"):
        assert save_key != "params", "params save_key is reserved"
        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return "{name}({args})".format(
            name=self.__class__.__name__, args=format_args(state)
        )

    def _get_target_function(self, key):
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, None)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        return params

    @property
    def target_dependence(self):
        return {}

    def add_targets(self, additional_targets):
        """Add targets to transform them the same way as one of existing targets
        ex: {'normals1': 'normals', 'normals2': 'normals'}

        Args:
            additional_targets (dict): keys - new target name, values -
            old target name. ex: {'normals2': 'normals'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class "
            + self.__class__.__name__
        )

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def get_transform_init_args_names(self):
        raise NotImplementedError(
            "Class {name} is not serializable because the `get_transform_init_args_names` method is not "
            "implemented".format(name=self.get_class_fullname())
        )

    def get_base_init_args(self):
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self):
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def _to_dict(self):
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self):
        d = self._to_dict()
        d["id"] = id(self)
        return d


class PointCloudsTransform(BasicTransform):
    """Transform for point clouds."""

    @property
    def targets(self):
        return {
            "points": self.apply,
            "normals": self.apply_to_normals,
            "features": self.apply_to_features,
            "cameras": self.apply_to_camera,
            "bbox": self.apply_to_bboxes,
            "labels": self.apply_to_labels,
        }

    def apply_to_bboxes(self, bboxes, **params):
        return [self.apply_to_bbox(bbox, **params) for bbox in bboxes]

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError(
            "Method apply_to_bbox is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_cameras(self, cameras, **params):
        return [self.apply_to_bbox(camera, **params) for camera in cameras]

    def apply_to_camera(self, camera, **params):
        raise NotImplementedError(
            "Method apply_to_camera is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_normals(self, normals, **params):
        raise NotImplementedError(
            "Method apply_to_normals is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_features(self, features, **params):
        raise NotImplementedError(
            "Method apply_to_features is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_labels(self, labels, **params):
        raise NotImplementedError(
            "Method apply_to_labels is not implemented in class "
            + self.__class__.__name__
        )


class NoOp(PointCloudsTransform):
    """Does nothing"""

    def apply(self, points, **params):
        return points

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_camera(self, camera, **params):
        return camera

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args_names(self):
        return ()








class Scale3d(PointCloudsTransform):
    """Scale the input point cloud.

    Args:
        scale_limit (float, float, float): maximum scaling of input point cloud.
            Default: (0.1, 0.1, 0.1).
        bias (list(float, float, float)): base scaling that is always applied.
            List of 3 values to determine the basic scaling. Default: (1, 1, 1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self, scale_limit=(0.1, 0.1, 0.1), bias=(1, 1, 1), always_apply=False, p=0.5
    ):
        super().__init__(always_apply, p)
        self.scale_limit = []
        for limit, bias_for_axis in zip(scale_limit, bias):
            self.scale_limit.append(to_tuple(limit, bias=bias_for_axis))

    def get_params(self):
        scale = []
        for limit in self.scale_limit:
            scale.append(random.uniform(limit[0], limit[1]))
        return {"scale": scale}

    def apply(self, points, scale=(1, 1, 1), **params):
        return F.scale(points, scale)

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {"scale_limit": self.scale_limit}


class RotateAroundAxis3d(PointCloudsTransform):
    """Rotate point cloud around axis on random angle.

    Args:
        rotation_limit (float): maximum rotation of the input point cloud. Default: (pi / 2).
        axis (list(float, float, float)): axis around which the point cloud is rotated. Default: (0, 0, 1).
        center_point (float, float, float): point around which to rotate. Default: mean points.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self,
        rotation_limit=math.pi / 2,
        axis=(0, 0, 1),
        center_point=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.rotation_limit = to_tuple(rotation_limit, bias=0)
        self.axis = axis
        self.center_point = center_point

    def get_params(self):
        angle = random.uniform(self.rotation_limit[0], self.rotation_limit[1])
        return {"angle": angle, "axis": self.axis, "center_point": self.center_point}

    def apply(self, points, axis, angle, **params):
        return F.rotate_around_axis(points, axis, angle, center_point=self.center_point)

    def apply_to_normals(self, normals, axis, angle, **params):
        return F.rotate_around_axis(normals, axis, angle, center_point=(0, 0, 0))

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {
            "rotation_limit": to_tuple(self.rotation_limit, bias=0),
            "axis": self.axis,
        }


class Crop3d(PointCloudsTransform):
    """Crop region from image.

    Args:
        x_min (float): Minimum x coordinate.
        y_min (float): Minimum y coordinate.
        z_min (float): Minimum z coordinate.
        x_max (float): Maximum x coordinate.
        y_max (float): Maximum y coordinate.
        z_max (float): Maximum z coordinate.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self,
        x_min=-math.inf,
        y_min=-math.inf,
        z_min=-math.inf,
        x_max=math.inf,
        y_max=math.inf,
        z_max=math.inf,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    @property
    def targets_as_params(self):
        return ["points"]

    def get_params_dependent_on_targets(self, params):
        return {
            "indexes": F.crop(
                params["points"],
                x_min=self.x_min,
                y_min=self.y_min,
                z_min=self.z_min,
                x_max=self.x_max,
                y_max=self.y_max,
                z_max=self.z_max,
            )
        }

    def apply(self, points, indexes, **params):
        return points[indexes]

    def apply_to_normals(self, normals, indexes, **params):
        return normals[indexes]

    def apply_to_labels(self, labels, indexes, **params):
        return labels[indexes]

    def apply_to_features(self, features, indexes, **params):
        return features[indexes]

    def get_transform_init_args_names(self):
        return ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")


class Center3d(PointCloudsTransform):
    """Move average of point cloud and move it to coordinate (0,0,0).

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(self, offset=(0, 0, 0), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.offset = offset

    def apply(self, points, **params):
        return F.move(F.center(points), self.offset)

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {
            "offset": self.offset,
        }


class Move3d(PointCloudsTransform):
    """Move point cloud on offset.

    Args:
        offset (float): coorinate where to move origin of coordinate frame. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(self, offset=(0, 0, 0), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.offset = offset

    def get_params(self):
        return {"offset": self.offset}

    def apply(self, points, offset, **params):
        return F.move(points, offset)

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {
            "offset": self.offset,
        }


class RandomMove3d(Move3d):
    """Move point cloud on random offset.

    Args:
        x_min (float): Minimum x coordinate. Default: -1.
        y_min (float): Minimum y coordinate. Default: -1.
        z_min (float): Minimum z coordinate. Default: -1.
        x_max (float): Maximum x coordinate. Default: 1.
        y_max (float): Maximum y coordinate. Default: 1.
        z_max (float): Maximum z coordinate. Default: 1.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self,
        x_min=-1.0,
        y_min=-1.0,
        z_min=-1.0,
        x_max=1.0,
        y_max=1.0,
        z_max=1.0,
        offset=(0, 0, 0),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(offset, always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def get_params(self):
        offset = [
            random.uniform(self.x_min, self.x_max),
            random.uniform(self.y_min, self.y_max),
            random.uniform(self.z_min, self.z_max),
        ]
        return {"offset": offset}

    def get_transform_init_args_names(self):
        return {
            "offset": self.offset,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "z_min": self.z_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "z_max": self.z_max,
        }


class RandomDropout3d(PointCloudsTransform):
    """Randomly drop points from point cloud.

    Args:
        dropout_ratio (float): Percent of points to drop. Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(self, dropout_ratio=0.2, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.dropout_ratio = dropout_ratio

    @property
    def targets_as_params(self):
        return ["points"]

    def get_params_dependent_on_targets(self, params):
        points_len = len(params["points"])
        indexes = random.sample(
            range(points_len), k=int(points_len * (1 - self.dropout_ratio))
        )
        sorted_indexes = sorted(indexes)
        return {"indexes": sorted_indexes}

    def apply(self, points, indexes, **params):
        return points[indexes]

    def apply_to_normals(self, normals, indexes, **params):
        return normals[indexes]

    def apply_to_labels(self, labels, indexes, **params):
        return labels[indexes]

    def apply_to_features(self, features, indexes, **params):
        return features[indexes]

    def get_transform_init_args(self):
        return {"dropout_ratio": self.dropout_ratio}


class Flip3d(PointCloudsTransform):
    """Flip point cloud around axis
        Implemented as rotation on 180 deg around axis.

    Args:
        axis (list(float, float, float)): Axis to flip the point cloud around. Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(self, axis=(1, 0, 0), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axis = axis

    def apply(self, points, **params):
        points = F.flip_coordinates(points, self.axis)
        return points

    def apply_to_normals(self, normals, **params):
        normals[:, self.axis] = -normals[:, self.axis]
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {"axis": self.axis}