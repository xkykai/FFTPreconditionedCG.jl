using MPI
MPI.Init()
comm = MPI.COMM_WORLD
println("rank=", MPI.Comm_rank(comm), " size=", MPI.Comm_size(comm),
        " lib=", basename(MPI.libmpi))
MPI.Finalize()
