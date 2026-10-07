"""Register the reference modules in memory; do not modify installed source files."""
import inspect
import textwrap


def register():
    import ultralytics
    import ultralytics.nn.tasks as tasks
    from hgdefect_yolo_modules import PPAPEMA, KDEHyperGraphFusion, CLAGRGCUFuse
    if getattr(tasks.parse_model, '_paper_formula_registered', False):
        return
    if ultralytics.__version__ != '8.2.103':
        raise RuntimeError('This integration is checked for ultralytics==8.2.103')
    source = textwrap.dedent(inspect.getsource(tasks.parse_model))
    anchor = '        elif m is AIFI:'
    if source.count(anchor) != 1:
        raise RuntimeError('Unsupported parse_model source; no files were changed')
    rules = '''        elif m is PPAPEMA:
            c2 = ch[f]
            args = [c2, *args]
        elif m is KDEHyperGraphFusion:
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [[ch[x] for x in f], c2, *args[1:]]
        elif m is CLAGRGCUFuse:
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [[ch[x] for x in f], c2, *args[1:]]
'''
    namespace = dict(tasks.__dict__)
    namespace.update(PPAPEMA=PPAPEMA, KDEHyperGraphFusion=KDEHyperGraphFusion, CLAGRGCUFuse=CLAGRGCUFuse)
    exec(compile(source.replace(anchor, rules + anchor), '<paper_formula_parse_model>', 'exec'), namespace)
    tasks.parse_model = namespace['parse_model']
    tasks.parse_model._paper_formula_registered = True
