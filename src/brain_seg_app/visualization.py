import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def compute_simple_volumes(seg, dx, dy, dz):
    voxel_vol_cm3 = (dx * dy * dz) / 1_000
    ncr = np.sum(seg == 1) * voxel_vol_cm3
    ed  = np.sum(seg == 2) * voxel_vol_cm3
    et  = np.sum(seg == 3) * voxel_vol_cm3
    return {
        "NCR (cm³)":   round(ncr, 2),
        "ED (cm³)":    round(ed, 2),
        "ET (cm³)":    round(et, 2),
        "Total (cm³)": round(ncr + ed + et, 2),
    }


def volumes_from_nii(nii_path: str):
    """
    Convenience wrapper: load a NIfTI seg file and compute labelled volumes.
    """
    import nibabel as nib

    img        = nib.load(nii_path, mmap=False)
    seg        = img.get_fdata()
    dx, dy, dz = img.header.get_zooms()[:3]              

    return compute_simple_volumes(seg, dx, dy, dz)

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
    mode: str = "Above",
    opacity: float = 0.6,
    cmap: list[str] | str = px.colors.sequential.Hot,   # <-- palette here
    width: int = 800,
    height: int = 800,
) -> go.Figure:

    # ------------- build composite map & hover text (unchanged) -------------
    unc_combined = np.zeros_like(brain_slice, dtype=float)
    hover_text   = np.full(brain_slice.shape, "", dtype=object)

    for label, unc in unc_slices:
        mask = (unc <= threshold) if mode == "Below" else (unc >= threshold)
        if not np.any(mask):
            continue
        unc_combined = np.where(mask, np.maximum(unc_combined, unc), unc_combined)
        txt = np.vectorize(lambda x: f"{label}: {x:.3f}")(unc)
        hover_text = np.where(
            mask,
            np.where(hover_text == "", txt, hover_text + "<br>" + txt),
            hover_text,
        )

    # ----------------------------- plot -------------------------------------
    fig = go.Figure()

    # anatomy
    fig.add_trace(
        go.Heatmap(
            z=brain_slice,
            colorscale="gray",
            showscale=False,
            hoverinfo="skip",
            zsmooth="best",
        )
    )

    # uncertainty overlay – note: **no `colorscale` here**
    fig.add_trace(
        go.Heatmap(
            z=unc_combined,
            opacity=opacity,
            text=hover_text,
            hoverinfo="text",
            coloraxis="coloraxis",
            showscale=True,
        )
    )

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=20, b=80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed", scaleanchor="x", scaleratio=1),
        coloraxis=dict(                              # <- palette lives here
            colorscale=cmap,
            colorbar=dict(title="Uncertainty", len=0.8, thickness=15),
        ),
    )

    return fig