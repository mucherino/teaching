
# Some simple examples of structures in Julia
#
# AM

# Person (abstract type)
abstract type Person end

# imposing a property to all instances of Person
# -> they all need to have a name attribute!
function has_name(person::Person)
    hasproperty(person,:name)
end

# a method for all persons
function initials(person::Person)
   return person.name[1:1]
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

# overriding Base show function for teachers
function Base.show(io::IO,teacher::Teacher)
   print(io,"Teacher(",teacher.name,", #courses = ",numberOfCourses(teacher),")")
end

# a method for teachers
function numberOfCourses(teacher::Teacher) :: Int64
   return length(teacher.courses)
end

