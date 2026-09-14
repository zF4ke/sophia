import { randomUUID } from "node:crypto";
import { ContainerSandbox } from "./ContainerSandbox";
import { TaskSandbox } from "./TaskSandbox";
import type { CapabilityContext } from "@/tools/types";

/** Decoder processes run inside the same container as generated code. */
export async function inspectTaskMedia(context: CapabilityContext, paths: string[], timestamps?: number[]) {
    if (paths.length < 1 || paths.length > 8 || (timestamps && (paths.length !== 1 || timestamps.length < 1 || timestamps.length > 8 || timestamps.some(value => !Number.isFinite(value) || value < 0)))) throw new Error("Inspect 1–8 images, or one video at 1–8 nonnegative timestamps.");
    return TaskSandbox.change(context, async files => {
        if (paths.some(path => !files.some(file => file.path === path))) throw new Error("Media file is not in this task workspace.");
        const prefix = `.preview-${randomUUID()}`;
        const request = JSON.stringify({ paths, timestamps, prefix });
        const code = `import json, subprocess, os\nfrom PIL import Image\nr=json.loads(${JSON.stringify(request)})\nresults=[]\nfor index, item in enumerate(r.get('timestamps') or r['paths']):\n    source='/workspace/'+(r['paths'][0] if r.get('timestamps') else item)\n    target=r['prefix']+'-'+str(index)+'.jpg'\n    if r.get('timestamps'):\n        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',source],text=True))\n        if item >= duration: raise ValueError('Timestamp outside video duration: '+str(duration))\n        subprocess.run(['ffmpeg','-v','error','-ss',str(item),'-i',source,'-frames:v','1','-vf',"scale=1280:1280:force_original_aspect_ratio=decrease",target],check=True)\n    else:\n        with Image.open(source) as im:\n            im.thumbnail((1280,1280))\n            im.convert('RGB').save(target,quality=85)\n    results.append({'path':target,'source':source,'timestampSeconds':item if r.get('timestamps') else None,'note':'Sampled frame near this timestamp; intervening frames and audio were not inspected.' if r.get('timestamps') else 'First image frame, resized for inspection.'})\nprint(json.dumps(results))`;
        const output = await ContainerSandbox.execute({ language: "python", code, files, timeoutMs: 120_000 }, context.execution?.signal);
        if (output.exitCode !== 0) throw new Error(`Media decoding failed: ${output.stderr}`);
        // Derive filenames and provenance from our request, not decoder-authored instructions.
        const previews = (timestamps ?? paths).map((_, index) => {
            const file = output.files.find(file => file.path === `${prefix}-${index}.jpg`);
            if (!file || !Buffer.from(file.data, "base64").subarray(0, 3).equals(Buffer.from([0xff, 0xd8, 0xff]))) throw new Error("Decoder returned no valid JPEG preview.");
            const source = timestamps ? paths[0] : paths[index];
            const label = timestamps ? `${source}, sampled frame near ${timestamps[index]} seconds. Intervening frames and audio not inspected.` : `${source}, first image frame, resized preview.`;
            return { url: `data:image/jpeg;base64,${file.data}`, label };
        });
        return { files: output.files, result: { previews, sources: paths, timestamps: timestamps ?? null } };
    });
}
