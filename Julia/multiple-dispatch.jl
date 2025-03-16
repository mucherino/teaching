
# An example of multiple dispatch in Julia
#
# AM

# Person (abstract type)
abstract type Person end

# property for all instances of Person
function has_name(person::Person)
   hasproperty(person,:name)
end

# Kid
struct Kid <: Person
   name::String
   schoolYear::Int64
   schoolName::String
end

# Teacher
struct Teacher <: Person
   name::String
   courses::Vector{String}
end

# Scientist
struct Scientist <: Person
   name::String
   research_fields::Vector{String}
end

# Computer (abstract type)
abstract type Computer end

# property for all instances of Computer
function has_power(computer::Computer)
   hasproperty(computer,:power)
end

# GamingConsole
struct GamingConsole <: Computer
   power::Int64
   battery_life::Int64
end

# Computer with GPU
struct GPU <: Computer
   power::Int64
   number_of_threads::Int64
end

# Quantum computer
struct Quantum <: Computer
   power::Int64
   number_of_qbits::Int64
end

# Kids only use GamingConsole structures
function work(kid::Kid,console::GamingConsole)
   print(kid.name," is playing with a Gaming Console")
end

# Teachers use Computer devices for teaching
function work(teacher::Teacher,computer::Computer)
   print(teacher.name," is teaching how to program a ",typeof(computer)," computer")
end

# Teaching quantum computing requires some quantum physics
function work(teacher::Teacher,computer::Quantum)
   println(teacher.name, " is introducing the basis of quantum physics")
   print("Programming on a ",typeof(computer)," computer requires this introduction")
end

# Scientists use Computer devices for doing research
function work(scientist::Scientist,computer::Computer)
   print(scientist.name," is doing research on a ",typeof(computer)," computer")
end

