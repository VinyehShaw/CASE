# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Legacy import alias (required by the frozen shipped environment).
# The environment's patched ``verl`` imports the causal-KL module tree under a
# historical package name that predates this release; keep that name resolving
# to this very package (same module objects, no duplicates) so the environment
# works unchanged. Importing ``distill`` installs the alias.
import importlib
import importlib.abc
import importlib.util
import sys

_LEGACY_NAME = "easydistill"


def _translate(name):
    if name == _LEGACY_NAME or name.startswith(_LEGACY_NAME + "."):
        return "distill" + name[len(_LEGACY_NAME):]
    return None


class _LegacyAliasLoader(importlib.abc.Loader):
    def __init__(self, real_name):
        self.real_name = real_name

    def create_module(self, spec):
        mod = importlib.import_module(self.real_name)
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        real = _translate(fullname)
        if real is None:
            return None
        return importlib.util.spec_from_loader(fullname, _LegacyAliasLoader(real))


# Insert before PathFinder so legacy names never produce duplicate modules.
if not any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _LegacyAliasFinder())
