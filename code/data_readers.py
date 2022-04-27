def read(folder=None):
    if folder is None:
        import os
        folder = os.path.abspath(
                os.path.join(__file__, "../../data"))
    import uproot
    parts = ["signal", "background"]
    data = {}
    for part in parts:
        path = os.path.join(folder, part + ".root")
        root_file = uproot.open(path)
        events = root_file["Events"]
        for name in events.keys():
            # select only intresting branches
            if "recoGenJets" not in name or "AK8" in name:
                continue
            if "m_specific" not in name and "p4Polar" not in name:
                continue
            # some robusness to error
            try:
                values = events[name].array()
            except NotImplementedError:
                continue
            pretty_name = name.split(".")[-1]
            # be sure it exists
            data[pretty_name] = data.get(pretty_name, [])
            data[pretty_name].append(values)
    return parts, data

