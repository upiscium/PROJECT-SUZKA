"""R20 integration harness (IP-22; #411).

Test-side code only: nothing under ``suzka/`` imports this package.  Every
subject is a NEW genesis subject on a throwaway root; the harness never reads
or writes production state.  IP-22a: factory, guards, providers, control
channel / ``serve`` wrapper, destination-bound client, approval stand-in,
manifest.  IP-22b / 22c / 22d attach through ``experiments.r20.hooks``.
"""
