# OpenFPM Hackathon: Spring 2024

## OpenFPM Hackathon

We are pleased to announce the first OpenFPM Hackathon to take place February 9/10, 2023 at the Center for Systems Biology Dresden.

With OpenFPM developing into a global open-source project for scalable scientific computing, the community desires to meet in order to synchronize contributions, review and produce code, and get to know the core developers personally.

This 2-day in-person Hackathon is aimed at users, contributors, and developers. The core OpenFPM Developers will be present and available during the entire hackathon, so this is a great opportunity to get your ideas and wishes into the project and start close collaborations with them.

But the Hackathon also welcomes people with little or no C++ coding skills who are willing to use the two days to contribute towards the OpenFPM documentation, tutorials, or revamping the OpenFPM [web page](http://openfpm.mpi-cbg.de).

Additionally, we welcome potential new users and people who are interested in learning more about OpenFPM in order to decide whether to start using it. You will have all the expert users and developers on site to talk to and answer your questions, and you can implement a first own hands-on example implementation to get familiar with OpenFPM and its usage.

But most importantly, this serves as a kick-off meeting for building a global community around OpenFPM. Be part of it!

## Time and Location:
------------------

Location: Center for Systems Biology Dresden, Pfotenhauerstr. 108, D-01307 Dresden, Germany. Seminar Room Top Floor. For directions, please see [here](https://www.csbdresden.de/contact/how-to-get-here/).

Time: February 9 and 10, 2023 (2 days) with the following agenda:

Day 1 - Feb 9, 2023:

09.00 - 12.30 Topic discussion, code review, division of tasks
12.30 - 13.30 Lunch break (self-paid)
13.30 - 16.30 Active coding
16.30 - 18.00 Problem review & feedback

Day 2 - Feb 10, 2023:

09.00 - 12.00 Active coding
12.00 - 13.00 Lunch break (self-paid)
13.30 - 14.00 Preparing 10 min presentation
14.00 - 16.00 Plenary project presentation and discussion
16.00 - 17.30 Friday seminar & beer hour

## Travel logistics and fee:
-------------------------

Participation in the hackathon is free of charge. There is no registration fee.

Participants are expected to book and cover their travel arrangements by themselves. We can not provide financial or administrative support for travel to and from Dresden. Also, food during the hackathon is on self-pay basis. There is a canteen conveniently on site, and several affordable restaurant options in walking distance. Budget around 10 Euros for a meal.

You can find directions to CSBD [here](https://www.csbdresden.de/contact/how-to-get-here/
).

If you are arriving by train, both Dresden Main Station and Dresden Neustadt station are conveniently connected to our institute by tramway or bus.

If you are arriving by plane, the easiest option is to look for flights into Dresden International Airport. Intercontinental connections need to transit in Zurich, Munich, or Frankfurt. From Dresden airport, there are trains to the city center (travel time 10 min) or you can take a taxi cab for around 30 Euros. Alternatively, the airports of Berlin (BER) and Prague are also feasible options. From there, buses and/or train service runs to Dresden in about 90 minutes.

For accommodation, there are several hotels in walking distance from the institute, and many more in the city center with tramway service running to the institute.

We can recommend the following hotels in the vicinity:

*   [Hotel am Blauen Wunder](https://www.habw.de/start-eng)
*   [Hotel Andreas](https://www.hotel-andreas-dresden.de/)

Please do not hesitate to get in touch with us if you have questions or require assistance with hotel booking.

## Registration:
-------------

Everyone interested is welcome to attend, regardless if you are already an OpenFPM crack, a beginner, or not yet using it at all; regardless if you want to contribute code, learn for yourself, or work on documentation and web page. There is something for every skillset.

If you wish to attend, please send an e-mail to the organizer [Serhii Yaskovets](mailto:yaskovet@mpi-cbg.de), so we can have a tally and book rooms accordingly. This also applies to local applicants from CSBD itself or other Dresden institutions. Thanks!

Registration deadline: Jan 31, 2023.

## Proposed topics
---------------

Following are some ideas for topics people could work on. But of course, the ideal case is that participants come with their own topics and goals and take advantage of the presence of the OpenFPM Developers in order to get them realized and suggest changes or extensions to the library core. Feel free to e-mail your topic to the organizers ahead of time, or simply bring it along with you to the opening plenary!

Importantly, there are also topics, like writing documentation or redoing the OpenFPM web page, that do not require C++ coding skills. In this sense, the hackathon also welcomes participants wishing to work on these important topics.

Topic proposal collected so far:

*   Granular contact models that would enable OpenFPM to perform granular simulations either of Hookean (linear) or Hertzian (non-linear) dynamics. Implement both approaches on top of OpenFPM data structures and implement a client application for DEM simulation of granular matter.
*   Extend support for polynomial regression to n-dimensional data by adding a wrapper for the [minter](https://git.mpi-cbg.de/mosaic/software/math/minter) library. One can then use minter to obtain polynomial representations of data stored as OpenFPM grid or particle properties. Currently, there is a stub implementation only for particle properties. This should be extended to grids and subgrids with a uniform interface.
*   Rewrite the OpenFPM [web page](http://openfpm.mpi-cbg.de) to make it easier to navigate, more informative and up-to-date. Extend the existing primers and tutorials. Integrate with Read the Docs.
*   Integrate and wrap in OpenFPM the Adaptive Particle Representation library ([libAPR](https://github.com/AdaptiveParticles/LibAPR)) for multi-resolution and adaptive simulations.