---
layout: blog_page
permalink: /blog/digital-twins/
publish_date: '23.12.2023'
title: 'Digital Twins'
image: '/assets/images/tello_dt.png'
description: 'What are digital twins? Why digital twin is not a new technology? What made digital twins popular
recently? How can semantics help digital twins?'
---

# Digital Twins (of Physical Systems)

Date: 23.12.2023

**TL;DR:** With great simplicity, digital replica of a physical system with a bidirectional communication in between
is referred as Digital Twins. They are not new, they are not simulations, and they are not just an Internet of Things
(IoT) network. Digital Twins are becoming highly popular in the past 5 years, following the advancements in networking
technologies (5G), deep learning, and the involvement of semantic technologies in smart environments in general.
Semantics help with system/data modeling, interoperability, (implicit) relation extraction / learning, and reasoning in
digital twins.

**Table of Contents**

1. [<u>Defining Digital Twins</u>](#defining-digital-twins)

    1. [<u>Historical Background</u>](#historical-background----houston-we-have-a-problem--)
    2. [<u>Twinning Ratio</u>](#twinning-ratio)

2. [<u>The hype around Digital Twins</u>](#the-hype-around-digital-twins)

3. [<u>Digital Twins and Semantics</u>](#digital-twins-and-semantics)

4. [<u>References</u>](#references)

## Defining Digital Twins

... is hard as researchers came up with many definitions [1][6]. Proposing another definition that is a
synthesis of others is not helpful. Instead, focusing on the commonalities of existing definitions can help
understanding what they are. I see the following two as the most repeated points [2]:

1. A continuous digital 'replica' of a physical system, that reflects physical reality as exact as possible.
2. Bidirectional communication in between the physical system, aka Physical Twin (PT), and the digital twin.

The digital replica is created by interpreting the information about the physical system in the digital environment
in a way that is easy to store and process. This includes metadata about both electronic and non-electronic physical
parts of the system and its environment. It is often referred as *Digital Shadow (DS)*.

<p align="center">
    <img alt="digital twins illustration taken from paper [2]" src="/assets/images/tello_dt.png" width="95%">
    <br/>
    <span>Figure 1: Digital Twins illustration taken from [2]</span>
</p>

Figure 1 illustrates both digital and physical 'spaces' together with the actions taken in each space. Both measurements
and static metadata about the system is interpreted and the digital representation of the physical space is updated
based on this interpretation. This is the main point of the first communication step, from PT to digital twin.

The second is that, digital representation can be used to run analysis, simulations and applications, e.g., machine
learning applications, which result in a set of actions to be performed in the PT. This is based on the representation
of the physical space. These actions can be towards making the physical system run more efficiently, mitigating problems
in the physical system or just routine actions. This means that the digital twin can perform many tasks depending on the
application and it is not characterized by certain set of applications.

### Historical Background - 'Houston, we have a problem!'

For most people, it is very surprising to hear that first digital twin, although it wasn't called a digital twin
at that time, is developed during the Apollo 13 mission, in 1970 [8]. The famous phrase "Houston, we have a problem"
from Jim Lovell, was actually an emergency call from physical space to the digital space.

After an explosion in one of the oxygen tanks inside the service module of Apollo 13, astronauts contacted the mission
control to inform them. The problem is diagnosed by the mission control thanks to the telemetry data that was
continuously being sent to them. In order to find a new configuration that can bring the astronauts safely back earth, a
simulation of the damaged spacecraft is run by the mission control.

This example still fits to the 2 common points of existing digital twin definitions described in the first part. There
is digital representation of the physical environment, which is how the mission control knew what was happening in the
spacecraft. And there is a bidirectional communication in between them, meaning that the mission control was able to
help re-configure the spacecraft using the data received and simulations.

<p align="center">
    <img alt="apollo 13 the first digital twin" src="/assets/images/apollo13.jpeg" width="95%">
    <br/>
    <span>Figure 2: Apollo 13 landing. Image credit: NASA.</span>
</p>

The next time when the digital twins are used without mentioning their current name is when Dr. Michael Grieves proposed
a conceptual ideal for Product Lifecycle Management (PLM) [5], in 2002 for manufacturing domain: "PLM is an integrated,
information driven approach to all aspects of a product’s life from its design inception, through its manufacture,
deployment and maintenance, and culminating in its removal from service and final disposal". In his presentation [5],
Dr. Grieves clearly shows the separation of physical and digital space, and the bidirectional communication in between
them.

### Twinning Ratio

Based on the mentioned use cases and definitions, it is clear that having a highly representative interpretation of the
physical space is crucial for digital twins. The more accurate the representation is, the more we can get out of digital
twins. However, there is no quantitative method to measure the 'representativeness', this is what we call in our
publication [3] as the *"twinning ratio"*.

We argued in our research that closer the digital representation to the PT is, more accurate the applications
running on the digital twin will be. Meaning that if we fail to capture a certain part of reality in the digital space,
our
applications using this digital representation will be less accurate. We call this *twinning ratio*, a qualitative
term that emphasizes the need for high 'enough' accuracy of the digital representation.

## The hype around Digital Twins

Many of the surveys around digital twins [1][4] show that they are becoming more and more popular. To understand the
reason, we can go back to the 2 initially mentioned common distinctive features of the digital twins; i) an accurate
'enough' continuous representation of the digital twin, ii) and a bidirectional communication in between them.

To establish an accurate continuous representation, we need the following 2 key technologies. The first is to have a
semantic model of the digital twin together with the concepts from the physical environment. Recent survey [1] shows
that researchers started working on this in the past 5 years only. Secondly, a fast and reliable sensor communication
system to get updates from the physical environment. I believe this is also happening in the past 5 (or slightly more)
years thanks to the advancements around 4G and 5G.

## Digital Twins and Semantics

Physical systems and their environment can get highly complex. Semantic technologies such as ontologies and knowledge
graphs can help modeling and implementing such complex systems. Ontologies however, can now be created much faster
thanks to Large Language Models (LLMs) [9]. Although LLMs can not create an ontology from scratch, they can help ease
the process enormously.

Besides system/data modeling, semantics can also help with interoperability, inferring implicit relations in between
physical components, and facilitates learning and reasoning tasks [3,4]. Semantic interoperability makes data machine
processable in and across digital twins. Using the semantic representation of the data, classical relation extraction
algorithms and/or machine learning-based algorithms can be run to infer relations in between physical system
subcomponents. Extracted/learned relations can then be used for semantic reasoning to make decisions on the physical
system.

### References

[1] R. D. D’Amico, J. A. Erkoyuncu, S. Addepalli, S.
Penver, <a target="_blank" href="https://www.sciencedirect.com/science/article/pii/S1755581722001158">Cognitive digital
twin: An approach to improve the maintenance management, CIRP Journal of Manufacturing Science and Technology</a> 38
(2022) 613–630.

[2] A. Tello, V. Degeler, <a target="_blank" href="https://pure.uva.nl/ws/files/106231123/22digitaltwins.pdf">Digital
Twins: An enabler for digital transformation, in: The Digital Transformation handbook</a>, Groningen Digital Business
Centre (GDBC), 2022. doi:10.5281/zenodo.7647493

[3] Karabulut, Erkan, Degeler, Victoria, and Groth, Paul. <a target="_blank" href="https://arxiv.org/abs/2310.07348">
Semantic Association Rule Learning from Time Series Data and Knowledge Graphs.</a> In SemIIM’23: 2nd International
Workshop on Semantic Industrial Information Modelling co-located with 22nd International Semantic Web Conference (ISWC
2023).

[4] Karabulut, Erkan, Salvatore F. Pileggi, Paul Groth, and Victoria
Degeler. <a target="_blank" href="https://www.sciencedirect.com/science/article/pii/S0167739X23004739">Ontologies in
digital twins: A systematic literature review.</a> Future Generation Computer Systems (2023).

[5] M.Grieves,
<a target="_blank" href="https://www.researchgate.net/publication/356192963_SME_Management_Forum_Completing_the_Cycle_Using_PLM_Information_in_the_Sales_and_Service_Functions">
Completing the cycle: Using plm information in the sales and service functions</a>. Conference: SME Management Forum
(10 2002).

[6] Barricelli, Barbara Rita, Elena Casiraghi, and Daniela
Fogli. <a target="_blank" href="https://ieeexplore.ieee.org/abstract/document/8901113"> A survey on digital twin:
Definitions, characteristics, applications, and design implications.</a> IEEE access 7 (2019): 167653-167671.

[7] NASA. Houston, we've got a problem. Accessed: November 17, 2023. Available
online: <a href="https://history.nasa.gov/ep76.pdf" target="_blank">Link.</a>

[8] Stephen Ferguson. "Apollo 13: The First Digital Twin." Available online:
<a target="_blank" href="https://blogs.sw.siemens.com/simcenter/apollo-13-the-first-digital-twin/">Link.</a>
Accessed: 23.12.2023.

[9] Babaei Giglou, Hamed, Jennifer D’Souza, and Sören
Auer. <a target="_blank" href="https://link.springer.com/chapter/10.1007/978-3-031-47240-4_22">
LLMs4OL: Large Language Models for Ontology Learning.</a> International Semantic Web Conference. Cham: Springer Nature
Switzerland, 2023.