import argparse
import json
import re

# Mapping from operator names (JSON) to actor IDs (XML)
name_to_actor_id = {
    "video_replayer": 1,
    "out_of_body_preprocessor": 3,
    "out_of_body_inference": 5,
    "out_of_body_postprocessor": 7

}

def load_wcet_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    updates = {}
    for name, values in data.items():
        if name in name_to_actor_id:
            actor_id = name_to_actor_id[name]
            low = int(values["min"])
            high = int(values["max"])
            updates[actor_id] = (low, high)
        else:
            print(f"Warning: Name '{name}' in JSON not found in name_to_actor_id mapping. Skipping.")
    return updates

def update_wcet_in_xml(xml_path, wcet_updates, output_path=None):
    if output_path is None:
        output_path = xml_path

    with open(xml_path, "r") as f:
        content = f.read()

    def replacer(match):
        actor_num = int(match.group(1))
        comment = match.group(3) or ''
        if actor_num in wcet_updates:
            low, high = wcet_updates[actor_num]
            return f"const int LOW_{actor_num} = {low}, HIGH_{actor_num} = {high};{comment}"
        else:
            return match.group(0)

    pattern = r"const int LOW_(\d+) = \d+, HIGH_\1 = \d+;(.*?)(\s*//.*)?"
    updated_content = re.sub(pattern, replacer, content)

    with open(output_path, "w") as f:
        f.write(updated_content)

    print(f"WCET values updated in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update WCET values in UPPAAL XML based on JSON mapping or manual set.")
    parser.add_argument("xml_path", help="Path to input XML file")
    parser.add_argument("-o", "--output", help="Output path to save updated XML")
    parser.add_argument("--json", help="Path to JSON file with WCET mappings")
    parser.add_argument("--set", nargs="+", help="Manual WCET changes actor:LOW,HIGH (e.g. 1:0,500)")

    args = parser.parse_args()

    if args.json:
        wcet_updates = load_wcet_from_json(args.json)
    elif args.set:
        wcet_updates = {}
        for arg in args.set:
            try:
                actor_id, values = arg.split(":")
                low, high = map(int, values.split(","))
                wcet_updates[int(actor_id)] = (low, high)
            except ValueError:
                raise ValueError(f"Invalid format for --set: '{arg}'. Use format like 1:0,500")
    else:
        raise ValueError("Either --json or --set must be provided to specify WCET updates.")

    update_wcet_in_xml(args.xml_path, wcet_updates, args.output)
