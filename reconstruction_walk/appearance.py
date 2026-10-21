"""Session-only material selection for the reconstructed visual mesh."""


class MeshAppearance:
    def __init__(self, stage, colored=False):
        from pxr import Usd, UsdShade, Sdf
        self.stage = stage
        mesh = stage.GetPrimAtPath("/World/WalkEnvironment/Visual/Surface")
        if not mesh:
            raise RuntimeError("Reconstructed visual mesh is missing")
        self.binding = UsdShade.MaterialBindingAPI(mesh)
        self.original_material = self.binding.ComputeBoundMaterial()[0]
        if not self.original_material:
            raise RuntimeError("Reconstructed visual mesh has no source material")
        # All overrides live in the session layer: cached USDs and source PLYs
        # retain their original vertex colors and material bindings.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            self.neutral_material = UsdShade.Material.Define(stage, "/World/WalkEnvironment/NeutralMaterial")
            shader = UsdShade.Shader.Define(stage, "/World/WalkEnvironment/NeutralMaterial/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((.55, .55, .55))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.85)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.)
            self.neutral_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.set_colored(colored)

    def set_colored(self, colored):
        from pxr import Usd
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.binding.Bind(self.original_material if colored else self.neutral_material)
        self.colored = bool(colored)
