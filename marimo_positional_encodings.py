import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from math import sin, cos, pow
    import altair as alt
    import pandas as pd
    import numpy as np
    return alt, cos, mo, np, pd, pow, sin


@app.cell
def _():
    d_model = 128
    max_pos = 128

    # i_dim = mo.ui.slider(1, max_pos, label="dim")
    # i_dim
    return d_model, max_pos


@app.cell
def _(alt, d_model, max_pos, np, pd):
    positions = np.arange(1, max_pos + 1)
    dimensions = np.arange(1, d_model + 1)
    grid_pos, grid_dim = np.meshgrid(positions, dimensions)
    pe_sin = np.sin(grid_pos / np.power(10000, 2*grid_dim/d_model))
    pe_cos = np.cos(grid_pos / np.power(10000, 2*grid_dim/d_model))
    pe = np.where(grid_dim % 2 == 0, pe_sin, pe_cos)
    df = pd.DataFrame({
        "pos": grid_pos.ravel(),
        "dim": grid_dim.ravel(),
        "pe": pe.ravel(),
    })
    alt.Chart(df).mark_rect().encode(
        x=alt.X("pos:O"),
        y=alt.Y("dim:O", scale=alt.Scale(reverse=True)),
        color=alt.Color("pe:Q"),
    # ).properties(
    #     height=500,
    #     width=500,
    )
    return df, dimensions, grid_dim, grid_pos, pe, pe_cos, pe_sin, positions


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
