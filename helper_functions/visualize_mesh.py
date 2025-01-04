import plotly.graph_objects as go
import numpy as np

def plot_3d_mesh(mesh, color='lightblue', opacity=0.50, title="3D Mesh Plot"):
    """
    Plots a 3D mesh using Plotly.
    
    Parameters:
        mesh: An object with `vertices` and `triangles` attributes (numpy arrays).
        color: The color of the mesh surface.
        opacity: The opacity level of the mesh surface (0 to 1).
        title: Title of the plot.
    """
    # Extract vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Create a 3D plot using Plotly
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color=color,
            opacity=opacity
        )
    ])

    # Update layout with axes titles and plot title
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        title=title
    )

    # Show the plot
    fig.show()

# Example Usage
# Assuming `mesh` is defined with `vertices` and `triangles` attributes
# plot_3d_mesh(mesh, color='lightblue', opacity=0.5, title="Stanford Bunny")

