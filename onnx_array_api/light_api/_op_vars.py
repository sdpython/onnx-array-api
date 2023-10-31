class OpsVars:
    """
    Operators taking multiple inputs.
    """

    def Add(self) -> "Var":
        self._check_nin(2)
        return self.make_node("Add", *self.vars_)

    def Reshape(self) -> "Var":
        self._check_nin(2)
        return self.make_node("Reshape", *self.vars_)
