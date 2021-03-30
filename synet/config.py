from synet.measures import overlap_parameter
from synet.measures import agent_entropy
from synet.measures import mixing_entropy
from synet.measures import path_entropy
from synet.measures import paint_entropy

measures = {
    "overlap": overlap_parameter,
    "agent": agent_entropy,
    "mixing": mixing_entropy,
    "path": path_entropy,
    "paint": paint_entropy,
}
