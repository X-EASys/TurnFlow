"""Allow only registered redundant T4 targets; all consumed facts stay strict."""

def make_parser(legacy_runtime):
    original = legacy_runtime.parse_response
    def parse_response(raw, spec):
        if spec.get('task') == 'T4' and spec.get('arm') == 'PROGRAM' and spec.get('derived_fields'):
            return original(raw, spec, allow_registered_t4_targets=True)
        return original(raw, spec)
    return parse_response
