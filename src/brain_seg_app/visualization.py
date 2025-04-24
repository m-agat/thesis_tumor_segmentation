import numpy as np
import plotly.graph_objects as go


def compute_simple_volumes(seg_data: np.ndarray, dx: float, dy: float, dz: float):
    """
    Compute the total volumes (in cm³) for each label in a segmentation.

    Parameters:
    -----------
    seg_data : np.ndarray
        A 3D array of integer labels, e.g. {0,1,2,3}.
    dx, dy, dz : float
        The voxel spacing (in mm) along each dimension.

    Returns:
    -------
    Tuple[float, float, float]
        Volumes for labels 1, 2, and 3 in cubic centimeters.
    """
    voxel_vol_mm3 = dx * dy * dz
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    ncr_voxels = np.sum(seg_data == 1)
    ed_voxels  = np.sum(seg_data == 2)
    et_voxels  = np.sum(seg_data == 3)

    ncr_volume = ncr_voxels * voxel_vol_cm3
    ed_volume  = ed_voxels  * voxel_vol_cm3
    et_volume  = et_voxels  * voxel_vol_cm3

    return ncr_volume, ed_volume, et_volume


def plot_segmentation(
    brain_slice: np.ndarray,
    seg_slice: np.ndarray,
    selected_tissues: list[tuple[int, tuple[float,float,float]]],
    opacity: float = 0.4,
    width: int = 800,
    height: int = 800
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=brain_slice,
        colorscale='gray',
        showscale=False,
        hoverinfo='skip',
        zsmooth='best'
    ))

    label_names = {1: 'Necrotic Core', 2: 'Edema', 3: 'Enhancing Tumor'}
    selected_tissues = sorted(selected_tissues, key=lambda x: x[0])
    composite_text = np.full(brain_slice.shape, '', dtype='<U50')

    for label_val, color in selected_tissues:
        mask = (seg_slice == label_val)
        if not np.any(mask):
            continue
        z_data = mask.astype(float)
        r,g,b = color
        rgba = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},1)'
        fig.add_trace(go.Heatmap(
            z=z_data,
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, rgba]],
            opacity=opacity,
            hoverinfo='skip',
            showscale=False
        ))
        composite_text[mask] = label_names.get(label_val, f'Tissue {label_val}')

    fig.add_trace(go.Heatmap(
        z=brain_slice,
        text=composite_text,
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',
        colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],
        opacity=1,
        showscale=False
    ))

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0,r=0,t=20,b=80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange='reversed', scaleanchor='x', scaleratio=1)
    )
    return fig


def plot_ground_truth(
    brain_slice: np.ndarray,
    gt_slice: np.ndarray,
    selected_gt_tissues: list[tuple[int, tuple[float,float,float]]],
    opacity: float = 0.4,
    width: int = 800,
    height: int = 800
) -> go.Figure:
    return plot_segmentation(brain_slice, gt_slice, selected_gt_tissues, opacity, width, height)


def plot_probabilities(
    brain_slice: np.ndarray,
    prob_slices: list[tuple[str, np.ndarray]],
    opacity: float = 0.4,
    width: int = 800,
    height: int = 800
) -> go.Figure:
    """
    Overlay softmax probability maps on a brain slice.

    prob_slices : list of (label, 2D prob array)
    """
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=brain_slice,
        colorscale='gray', showscale=False, hoverinfo='skip', zsmooth='best'
    ))

    color_map = {'NCR':(255,0,0),'ED':(0,255,0),'ET':(0,0,255)}
    composite_text = np.full(brain_slice.shape, '', dtype='<U200')

    for label, prob_slice in prob_slices:
        if not np.any(prob_slice):
            continue
        r,g,b = color_map.get(label, (255,255,255))
        rgba = f'rgba({r},{g},{b},1)'
        fig.add_trace(go.Heatmap(
            z=prob_slice,
            colorscale=[[0,'rgba(0,0,0,0)'],[1,rgba]],
            opacity=opacity,
            hoverinfo='skip',
            showscale=False
        ))
        fmt = np.vectorize(lambda p: f"{label} Prob: {p:.3f}")
        text_layer = fmt(prob_slice)
        first_mask = (composite_text == '')
        composite_text = np.where(
            first_mask,
            text_layer,
            np.char.add(np.char.add(composite_text, '<br>'), text_layer)
        )

    fig.add_trace(go.Heatmap(
        z=brain_slice,
        text=composite_text,
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',
        colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],
        opacity=1,
        showscale=False
    ))

    fig.update_layout(
        width=width, height=height,
        margin=dict(l=0,r=0,t=20,b=80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange='reversed', scaleanchor='x', scaleratio=1)
    )
    return fig


def plot_uncertainty(
    brain_slice: np.ndarray,
    unc_slices: list[tuple[str, np.ndarray]],
    threshold: float,
    mode: str,
    opacity: float = 0.4,
    width: int = 800,
    height: int = 800
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=brain_slice,
        colorscale='gray', showscale=False, hoverinfo='skip', zsmooth='best'
    ))

    unc_colors = {'NCR':(1,0,0), 'ED':(0,1,0), 'ET':(0,0,1)}
    composite_text = np.full(brain_slice.shape, '', dtype='<U50')

    for label, unc_data in unc_slices:
        if mode == 'Below':
            masked = np.where(unc_data <= threshold, unc_data, 0.0)
        else:
            masked = np.where(unc_data >= threshold, unc_data, 0.0)
        if not np.any(masked):
            continue
        r,g,b = unc_colors.get(label, (1,1,1))
        rgba = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},1)'
        fig.add_trace(go.Heatmap(
            z=masked,
            colorscale=[[0,'rgba(0,0,0,0)'],[1,rgba]],
            opacity=opacity,
            hoverinfo='skip',
            showscale=False
        ))
        fmt = np.vectorize(lambda x: f"{x:.3f}")
        txt = fmt(masked)
        mask_text = (composite_text == '')
        composite_text = np.where(
            mask_text,
            txt,
            np.char.add(np.char.add(composite_text, '<br>'), txt)
        )

    fig.add_trace(go.Heatmap(
        z=brain_slice,
        text=composite_text,
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',
        colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],
        opacity=1,
        showscale=False
    ))

    fig.update_layout(
        width=width, height=height,
        margin=dict(l=0,r=0,t=20,b=80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange='reversed', scaleanchor='x', scaleratio=1)
    )
    return fig
