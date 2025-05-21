def parse_annotations(file_path, output_path=None):
    captions = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            # Extract all objects and verb
            parts = line.strip().split('>')
            verb = parts[0].split('<')[1]
            objects = [part.split('<')[-1] for part in parts[1:-1]]
            objects = objects[0].split(',')
            frame_range = line.split('(')[1].split(')')[0]

            objects = [o.replace('water', 'water bottle').replace('chocolate', 'chocolate bottle').replace('peanut', 'peanut butter').replace('tea', 'tea bag') for o in objects]

            # Build verb phrase
            verb_conj = verb + 's'

            if verb in ['pour', 'put']:
                if len(objects) == 1:
                    action = f"{verb_conj} the {objects[0]}"
                elif len(objects) == 2:
                    action = f"{verb_conj} {objects[0]} onto {objects[1]}"
                else:
                    onto_obj = objects[1]
                    with_objs = ' with ' + ' and '.join(objects[2:])
                    action = f"{verb_conj} {objects[0]} on {onto_obj}{with_objs}"
            elif verb == 'stir':
                action = f"{verb_conj} the {' in the '.join(objects)}"
            elif verb == 'scoop':
                action = f"{verb_conj} the {' with the '.join(objects)}"
            elif verb == 'spread':
                action = f"{verb_conj} the {objects[0]} with the {objects[1]} on the {objects[2]}"

            else:
                action = f"{verb_conj} the {' and '.join(objects)}"

            caption = f"{frame_range}: The person {action}."
            captions.append(caption)

        except Exception as e:
            print(f"Skipping malformed line: {line.strip()} ({e})")

    # Write to output file if specified
    if output_path:
        with open(output_path, 'w') as f:
            for caption in captions:
                f.write(caption + '\n')
    else:
        for caption in captions:
            print(caption)

    return captions



from pathlib import Path

def list_txt_files(directory):
    txt_files = list(Path(directory).rglob('*.txt'))  # Use glob if you don't want to recurse
    return txt_files

# Example usage:
dir_paths = list_txt_files('data/gtea/labels')


# Example usage:
# Assume 'annotations.txt' contains the input lines
for p in dir_paths:
    parse_annotations(p, f'new_caps/{p.stem}_new.txt')

hi = 1