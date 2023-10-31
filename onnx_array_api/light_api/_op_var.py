class OpsVar:
    """
    Operators taking only one input.
    """

    def Neg(self) -> "Var":
        return self.make_node("Neg", self)
