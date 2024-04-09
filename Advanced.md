
# Advanced Programming

## Course organization

This course covers several alternative programming paradigms
and languages. By taking Java as an *intermediate* language, 
the main aim is to explore those languages that can be placed 
at lower level (such as C, [llvm](https://llvm.org), and even 
assembly), as well as others that can be placed at higher levels 
(such as the recent [Julia](https://julialang.org/)).

Our pedagogical approach consists of two main parts. Firstly,
a theoretical study on a selection of languages is planned, with 
a particular focus on their various paradigms, on their syntax, 
on their performance. This part of the work is partially presented 
by the teacher during the lectures, but students are also invited
to contribute by performing the first assignment detailed below.

Secondly, an extensive programming work is planned as well, where 
each group of students is supposed to develop one of the software 
projects listed below. The paradigm and the language used for the 
development of the chosen project are not predefined, it is up to 
each group to find out which ones better fit with each project. 
This work on the projects is referred to as the second assignment
in the text below.

This is an *advanced* programming course. The pedagogical
approach may be considered to be "advanced" as well, or if you 
prefer, "alternative". In this course, in order to get their grades, 
the students are not supposed to simply repeat to the teacher what 
they have learned from the lectures. Rather, they need to use (a 
selection of) the information they are able to get during the course 
to work in the best possible way on their assignments. In other words, 
the information does not simply need to be acquired, it's the 
use of the information that is actually going to be evaluated. 
The final grade takes into consideration the quality of the work 
that has been done for these two assignments. No additional 
(classical) exams are planned (unless some students can prove
that their absence to some of the activities are justified).

## Student groups and assignments

Students are free to choose their (unique) partner in order 
to form a working group on one of the two student assignments.
The only constraint is that every formed group is only allowed
to  work on one of the two assignments. In other words, if you 
are student A, you can form the group AB with student B to work 
on the first assignment, but you cannot work with the same 
student B on the second assignment. You'll have to find 
another student, say C, in order to form another group, 
say AC, and work on the second assignment.

The first assignment consists in exploring a new programming
language and in presenting its main features to the rest
of the class. The programming languages are assigned 
randomly to the groups, and chosen from a set of 
[languages supporting llvm](https://llvm.org/ProjectsWithLLVM/).
Each group can get inspiration on how to present the language 
assigned to them from the lectures that your teacher will give 
on *higher* and *lower* programming languages.

The second assignment consists in developing a software project. 
The initial lectures of the course are going to give some insights 
on some of the main topics considered in the projects. Before 
beginning with the practical sessions (in the lab), every group 
of students is supposed to prepare a short proposition where they 
motivate their choice of the project. The programming languages 
and software tools that they plan to use need to be pointed out as 
well, together with a preliminary working plan. It is recommended 
that each group of students writes a proposition for at least 2 of 
the projects listed below. It's the teacher who makes the final 
project assignment to each student group on the basis of the 
received propositions.

Once one of the projects is assigned to your group, please
follow the scheduled activities (see below) for every lab 
devoted to the projects (9 over 12).

## Lectures 

Please notice that the pedagogical material given below does
not completely cover the course content. Your presence at
the courses is highly recommended.

### Lectures on Julia programming language

Please follow [this link](./Julia/README.md).

### Lectures on low-level programming

Please follow [this link](./lowlevel/README.md).

### Lectures on comparisons among programming languages

Please follow [this link](./cmp/README.md).

### Lectures on project topics

- [Introduction to Distance Geometry](https://www.antoniomucherino.it/download/slides/DistanceGeometryIntroSlides.pdf);
- [Distance Geometry in 1D, subset sum and optic computing](https://www.antoniomucherino.it/download/slides/DistanceGeometry1D.pdf);
- [Feature Selection by Biclustering](https://www.antoniomucherino.it/download/slides/Valparaiso-charla2011.pdf).

Project topics not covered in these presentations are presented
without supporting slides. 

## List of programming languages (first assignment)

The assignment of the programming language to each
student group is completely random. Please check with
your teacher. The full list includes:

- [Crack](https://crack-lang.org/);
- [D](https://dlang.org/);
- [Emscripten](https://emscripten.org/);
- [Pony](https://www.ponylang.io/);
- [Pure](https://agraef.github.io/pure-lang/);
- [Terra](https://terralang.org/).

Our first lab sessions are devoted to the bibliographic search 
and testing of the programming language assigned to each groups.
Presentations of the programming languages (about 20 minutes long),
given by each group of students, will be scheduled during the semester.

## List of software projects (second assignment)

Nine of our 12 lab sessions are devoted to the software projects.
Recall that all groups of students are supposed to read *all* projects 
and write a short proposition (including a preliminary working plan) 
for at least two of them. The programming language that each group will 
select does not need to be one of the languages studied during the course.
The full list of projects includes:

- [Simulating Optical Circuits](https://www.antoniomucherino.it/download/PA/project1-optics.pdf);
- [Distance Geometry in 1D](https://www.antoniomucherino.it/download/PA/project2-dgp1.pdf);
- [Directed Graphs with Variable Vertex Orders](https://www.antoniomucherino.it/download/PA/project3-graph.pdf);
- [Hardness of Combinatorial Optimization Problems](https://www.antoniomucherino.it/download/PA/project4-hardness.pdf);
- [A Statistical Analysis on Human Motions](https://www.antoniomucherino.it/download/PA/project5-analysis.pdf);
- [Feature Selection by Biclustering](https://www.antoniomucherino.it/download/PA/project6-biclustering.pdf);
- [Combinatorial Optimization for Data Compression](https://www.antoniomucherino.it/download/PA/project7-compression.pdf);
- [The Distance Geometry Game](https://www.antoniomucherino.it/download/PA/project8-game.pdf).

## Planned lab work per session

Your presence to the lab sessions is very important for ensuring
a smooth development of the projects.

#### Lab1 *GitLab repository and bibliographic search*

Every group creates a new repository on our 
[GitLab](https://gitlab.istic.univ-rennes1.fr/)
and starts to write a README file where the preliminary ideas about 
your project are collected. Has anybody worked on a project similar 
to yours? Are there any existing implementations that may turn out to
be useful? Every group should include some relevant references in 
their README file, and cite these references in their text. The teacher 
needs to be invited as a "Reporter" of the repository.

#### Lab2 *Project initialization*

The final choice on the programming language needs to be done by the 
end of this lab, and at least one commit containing code needs to be 
pushed on the repository.

#### Lab3 *Development I*

The very first working version of the code (even if very preliminary)
should be pushed on the repository by the end of this lab session.

#### Lab4 *Development II*

Another push at the end of this lab is necessary. Normally, every group 
is supposed to have a code already performing some main tasks at this stage.

#### Lab5 *Testing*

By the end of this lab, each repository needs to contain a set of test 
instances each group is using for testing the developed code. Even if 
the code may still be under development, the testing phase can initially 
be performed on its working parts, and extended in a second step to the
entire project.

#### Lab6 *Finalization*

It is expected that the testing phase has implied some changes on the 
code, which each group is now supposed to push on the repository by the 
end of this lab. Make sure the programs are quite robust: cross corrections 
will begin soon!

#### Lab7 *Cross corrections I*

The teacher assigns a reviewer to each project (chosen among the students
that are not involved in the project). The reviewer needs to be included 
in quality of "Reporter" to the repository. By the end of the lab session, 
the reviewer needs to communicate to the teacher a short review on the 
README and other main information that are available on the repository.
Possible questions to answer are: Is the main information available and 
clear? Is the main contribution well stated?

#### Lab8 *Cross corrections II*

By the end of this lab, each reviewer is supposed to submit to the teacher 
an evaluation of the codes of the project assigned to them. Is the code 
well-written and clear? Well commented? Was the choice of the language
appropriate? Can one easily run tests? Can one easily run the code on new 
instances (not the ones used by the developers)?

#### Lab9 *Project revision*

Before the beginning of this lab, the teacher will have sent to each group 
a text containing the revision comments from the reviewers. The teacher may 
decide to filter out some of the original comments, and to add others. This 
last lab is devoted to the very last changes on the code, possibly by taking 
into consideration the received comments and suggestions.

## Links

* [Back to main repository page](README.md)

