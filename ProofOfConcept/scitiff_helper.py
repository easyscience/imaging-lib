import scipp as sc
import scitiff
from dataclasses import dataclass


class _FileNameLog:
    def __init__(self, filename: str) -> None:
        self._filename = filename

    def _repr_html_(self) -> str:
        return f"<h3>{self._filename}</h3>"


def _transform_position(da: sc.DataArray) -> sc.DataArray:
    x = da.coords["x"]
    y = da.coords["y"]
    z = da.coords["z"]
    z = z.broadcast(dims=["z"], shape=[1 if z.shape == () else z.shape[0]])
    length = x.shape[0] * y.shape[0] * z.shape[0]
    new_da = da.copy()

    def _wrap_pos(var: sc.Variable) -> sc.Variable:
        return (
            var.broadcast(
                dims=[var.dim, "position"], shape=[var.shape[0], length // var.shape[0]]
            ).flatten(dims=[var.dim, "position"], to="position")
            # .rename_dims({var.dim: 'element'})
        )

    xyz = sc.concat([_wrap_pos(x), _wrap_pos(y), _wrap_pos(z)], "pos").transpose(
        ["position", "pos"]
    )
    position = (
        sc.vectors(dims=["position"], values=xyz.values, unit=xyz.unit)
        .fold(dim="position", sizes={"x": x.shape[0], "y": y.shape[0], "z": z.shape[0]})
        .squeeze()
    )
    new_da.coords["pixel_position"] = position
    return new_da


def load_scitiff(filename, background: bool = False, show: bool = True) -> sc.DataArray:
    from ess.imaging.normalize import apply_threshold_to_background_image

    da = (
        scitiff.load_scitiff(filename)["image"]
        #.astype(int)
        .astype(float)
        .rename_dims({"t": "tof"})
    )
    variances = da["c", sc.scalar("variances")].data.values

    if background:
        da = apply_threshold_to_background_image(
            background=da, background_threshold=sc.scalar(1.0, unit="counts")
        )
    da = _transform_position(da)["c", 0].copy(deep=True)
    da.variances = variances

    if show:
        display(_FileNameLog(filename))
        display(da)

    return da


@dataclass
class ROI:
    x_b: sc.Variable
    x_t: sc.Variable
    y_b: sc.Variable
    y_t: sc.Variable

    def slice_dataarray(self, da: sc.DataArray) -> sc.Variable:
        """
        Convert the ROI to a variable slice
        """
        x = da.coords["x"]
        y = da.coords["y"]
        return da[x.dim, self.x_b : self.x_t][y.dim, self.y_b : self.y_t]

    def __hash__(self) -> int:
        return hash((self.x_b.value, self.x_t.value, self.y_b.value, self.y_t.value))


def make_rectangle_tool(normalized: sc.DataArray, roi_container: set):
    import plopp as pp
    from plopp.widgets.drawing import DrawingTool
    from functools import partial
    from mpltoolbox import Rectangles

    def vertical_sum(da, rect_info):
        """
        Function that slices the data according to the
        rectangle size/position, and sums along the
        vertical dimension.
        """
        x = rect_info["x"]
        y = rect_info["y"]
        b = min(y["bottom"], y["top"])
        t = max(y["bottom"], y["top"])
        l = min(x["left"], x["right"])
        r = max(x["left"], x["right"])
        roi_container.add((ROI(x_b=l, x_t=r, y_b=b, y_t=t)))
        return normalized[y["dim"], b:t][x["dim"], l:r].mean("x").mean("y")

    def _get_rect_info(artist, figure):
        """
        Convert the raw rectangle info to a dict containing the dimensions of
        each axis, and values with units.
        """
        return lambda: {
            "x": {
                "dim": figure.canvas.dims["x"],
                "left": sc.scalar(artist.xy[0], unit=figure.canvas.units["x"]),
                "right": sc.scalar(
                    artist.xy[0] + artist.width, unit=figure.canvas.units["x"]
                ),
            },
            "y": {
                "dim": figure.canvas.dims["y"],
                "bottom": sc.scalar(artist.xy[1], unit=figure.canvas.units["y"]),
                "top": sc.scalar(
                    artist.xy[1] + artist.height, unit=figure.canvas.units["y"]
                ),
            },
        }

    RectangleTool = partial(
        DrawingTool,
        tool=Rectangles,
        get_artist_info=_get_rect_info,
        icon="vector-square",
    )

    from plopp.widgets import HBar

    data_node = pp.Node(normalized.mean("tof"))

    f2d = pp.imagefigure(
        data_node,
        norm="log",
        title="Select ROIs here:\nAverage along TOF",
        aspect="equal",
    )
    f1d = pp.linefigure(title="Distribution in ROIs")

    r = RectangleTool(
        figure=f2d, input_node=data_node, func=vertical_sum, destination=f1d
    )
    f2d.toolbar["roi"] = r

    return HBar([f2d, f1d])
