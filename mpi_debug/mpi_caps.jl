using MPI
MPI.Init()
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("MPI_LIBRARY     = ", MPI.MPI_LIBRARY)
    println("library_version = ", strip(first(split(MPI.Get_library_version(), '\n'))))
    println("has_cuda        = ", MPI.has_cuda())
    maps = read("/proc/self/maps", String)
    for pat in ("libmpiwrapper", "hpcx", "libmpi.so.40", "libopen-pal", "libcuda")
        println("mapped $pat  = ", occursin(pat, maps))
    end
end
MPI.Finalize()
