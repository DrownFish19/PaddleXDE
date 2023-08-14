class ContainerMeta(type):
    def all(cls):
        return sorted(getattr(cls, x) for x in dir(cls) if not x.startswith("__"))

    def __str__(cls):
        return str(cls.all())

    def __contains__(cls, item):
        return item in cls.all()
