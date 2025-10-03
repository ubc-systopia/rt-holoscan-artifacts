import argparse
import re

def parse_wcet_args(wcet_args):
    updates = {}
    for arg in wcet_args:
        try:
            actor_id, values = arg.split(":")
            low, high = map(int, values.split(","))
            updates[int(actor_id)] = (low, high)
        except ValueError:
            raise ValueError(f"Invalid format for --set: '{arg}'. Use format like 1:0,500")
    return updates

def update_wcet_in_xml(xml_path, wcet_updates, output_path=None):
    if output_path is None:
        output_path = xml_path  # Overwrite original if not specified

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
    parser = argparse.ArgumentParser(description="Update WCET values in a UPPAAL XML file.")
    parser.add_argument("xml_path", help="Path to the input XML file")
    parser.add_argument("--set", nargs="+", help="WCET changes in format actor:LOW,HIGH (e.g. 1:0,500)", required=True)
    parser.add_argument("-o", "--output", help="Path to save the updated XML file")  # OPTIONAL

    args = parser.parse_args()
    wcet_updates = parse_wcet_args(args.set)
    update_wcet_in_xml(args.xml_path, wcet_updates, args.output)
