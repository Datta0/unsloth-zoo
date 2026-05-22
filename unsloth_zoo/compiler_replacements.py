# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "comment_out_model_decorators_enabled",
    "comment_out_model_torch_decorators",
    "compiler_replacements",
    "model_compile_decorator",
    "model_disable_decorator",
]

import os
import re

compiler_replacements = {}

def comment_out_model_decorators_enabled():
    return os.environ.get("UNSLOTH_COMPILE_COMMENT_OUT_MODEL_DECORATORS", "0") == "1"
pass

def model_compile_decorator(fullgraph):
    decorator = (
        f"@torch.compile(fullgraph = {fullgraph}, dynamic = True, "
        "options = torch_compile_options)"
    )
    if comment_out_model_decorators_enabled():
        decorator = "# " + decorator
    return decorator
pass

def model_disable_decorator():
    decorator = "@torch.compiler.disable(recursive = False)"
    if comment_out_model_decorators_enabled():
        decorator = "# " + decorator
    return decorator
pass

def comment_out_model_torch_decorators(source):
    if not comment_out_model_decorators_enabled():
        return source
    return re.sub(
        r"(?m)^([ \t]*)@(torch\.compile\([^\n]*\)|torch\.compiler\.disable\(recursive = False\))",
        r"\1# @\2",
        source,
    )
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
