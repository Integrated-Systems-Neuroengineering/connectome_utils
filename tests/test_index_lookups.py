"""get_neuron_by_idx, get_neuron_by_hbmIdx and get_axon_by_idx return exactly
what the old scans over connectomeDict returned (the first matching entry in
connectomeDict order, or None), on one core and on several, before and after
every way the connectome assigns or changes indices.

    python -m unittest tests.test_index_lookups
"""
import random
import sys
import types
import unittest

try:
    import jaal  # noqa: F401  (connectome.py imports it for graph_viz only)
except ImportError:
    sys.modules["jaal"] = types.SimpleNamespace(Jaal=None)

from connectome_utils.connectome import connectome, neuron


def scan(c, kind, value, idx):
    """The lookups as they were before the tables."""
    for key in c.connectomeDict:
        entry = c.connectomeDict[key]
        if entry.get_neuron_type() == kind and value(entry) == idx:
            return entry
    return None


LOOKUPS = {
    "get_neuron_by_idx": ("neuron", lambda e: e.get_coreTypeIdx()),
    "get_neuron_by_hbmIdx": ("neuron", lambda e: e.get_hbmIdx()),
    "get_axon_by_idx": ("axon", lambda e: e.get_coreTypeIdx()),
}


def network(n_axons=40, n_neurons=150, seed=0):
    neuron.reset_count()
    rng = random.Random(seed)
    c = connectome()
    axons = [neuron("a%d" % i, "axon", axonType="Uaxon") for i in range(n_axons)]
    cells = [neuron("n%d" % i, "neuron", neuronModel=rng.choice(["LIF_a", "LIF_b"]),
                    output=(i % 9 == 0)) for i in range(n_neurons)]
    for entry in axons + cells:
        c.addNeuron(entry)
    for a in axons:
        for j in rng.sample(range(n_neurons), 4):
            a.addSynapse(cells[j], rng.randint(-8, 8))
    for cell in cells:
        for j in rng.sample(range(n_neurons), 3):
            cell.addSynapse(cells[j], rng.randint(-8, 8))
    return c, rng


class IndexLookups(unittest.TestCase):
    def assertSameAsScan(self, c, extra=()):
        for name, (kind, value) in LOOKUPS.items():
            for idx in list(range(-2, 400)) + [None] + list(extra):
                self.assertIs(getattr(c, name)(idx), scan(c, kind, value, idx),
                              "%s(%r)" % (name, idx))

    def test_one_core(self):
        c, _ = network()
        self.assertSameAsScan(c)                    # global indices, no hbmIdx yet
        c.pad_models()
        c.get_class_ordered_list()
        self.assertSameAsScan(c)

    def test_several_cores_and_every_reassignment(self):
        c, rng = network(seed=1)
        self.assertSameAsScan(c)                    # tables built here
        c.apply_partition({k: rng.randrange(4) for k in c.connectomeDict})
        self.assertSameAsScan(c)                    # cores, relay axons, axon indices
        c.pad_models()
        self.assertSameAsScan(c)                    # pad neurons, per-core coreTypeIdx
        c.get_class_ordered_list()
        self.assertSameAsScan(c)                    # per-core hbmIdx
        cell = c.get_neuron_by_key("n5")
        cell.set_hbmIdx(9999)
        cell.set_coreTypeIdx(8888)
        cell.set_core(3)
        self.assertSameAsScan(c, extra=(8888, 9999))
        c.addNeuron(neuron("late", "neuron", neuronModel="LIF_a"))
        self.assertSameAsScan(c)
        for entry in c.connectomeDict.values():     # the spike lookup still agrees
            if entry.get_neuron_type() == "neuron" and entry.get_hbmIdx() is not None:
                self.assertIs(c.get_neuron_by_core_idx(entry.get_core(), entry.get_hbmIdx()),
                              entry)

    def test_an_earlier_entry_taking_an_index_wins_as_in_the_scan(self):
        """A cached answer can still look valid (it kept its index) after an
        EARLIER entry in connectomeDict takes the same index; the scan returns
        the earlier one, so every change must drop the tables."""
        c, rng = network(seed=4)
        c.apply_partition({k: rng.randrange(3) for k in c.connectomeDict})
        c.pad_models()
        c.get_class_ordered_list()
        # n0 is the first neuron in connectomeDict, so it wins every index it holds
        first, later = c.get_neuron_by_key("n0"), c.get_neuron_by_key("n40")
        own_idx = first.get_coreTypeIdx()
        later.set_hbmIdx(5000)
        self.assertIs(c.get_neuron_by_hbmIdx(5000), later)
        first.set_hbmIdx(5000)
        self.assertIs(c.get_neuron_by_hbmIdx(5000), first)
        later.set_coreTypeIdx(6000)
        self.assertIs(c.get_neuron_by_idx(6000), later)
        first.set_coreTypeIdx(6000)
        self.assertIs(c.get_neuron_by_idx(6000), first)
        self.assertSameAsScan(c)
        c.update_class_ordered_coreIdx()            # writes coreTypeIdx directly; n0 gets
        self.assertIs(c.get_neuron_by_idx(own_idx), first)  # own_idx back (a hit, no miss first)
        self.assertSameAsScan(c)
        later.set_hbmIdx(5001)                      # (core, hbmIdx) is unique in a real network
        core, idx = first.get_core(), first.get_hbmIdx()
        self.assertIs(c.get_neuron_by_core_idx(core, idx), first)
        first.set_core(core + 1)                    # the spike lookup moves with it
        self.assertIsNot(c.get_neuron_by_core_idx(core, idx), first)
        self.assertIs(c.get_neuron_by_core_idx(core + 1, idx), first)

    def test_an_index_written_without_a_setter_is_still_found(self):
        c, _ = network(seed=2)
        c.pad_models()
        c.get_class_ordered_list()
        self.assertSameAsScan(c)
        cell = c.get_neuron_by_key("n7")
        old = cell.get_hbmIdx()
        cell.hbmIdx = 7777                          # bypasses set_hbmIdx
        self.assertIs(c.get_neuron_by_hbmIdx(7777), cell)
        self.assertIs(c.get_neuron_by_hbmIdx(old), scan(c, "neuron", lambda e: e.get_hbmIdx(), old))

    def test_hits_reuse_one_table(self):
        c, _ = network(seed=3)
        c.pad_models()
        c.get_class_ordered_list()
        c.get_neuron_by_hbmIdx(0)
        table = c._lookup
        for idx in range(100):
            c.get_neuron_by_hbmIdx(idx)
            c.get_neuron_by_idx(idx)
        self.assertIs(c._lookup, table)


if __name__ == "__main__":
    unittest.main()
