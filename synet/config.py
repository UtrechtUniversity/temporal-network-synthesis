from synet.measures.overlap_parameter import OverlapParameter
from synet.measures.agent_entropy import AgentEntropy
from synet.measures.mixing import MixingEntropy
from synet.measures.paths import PathEntropy
from synet.measures.paint import PaintEntropy

measures = {
    "overlap": OverlapParameter,
    "agent": AgentEntropy,
    "mixing": MixingEntropy,
    "path": PathEntropy,
    "paint": PaintEntropy,
}
