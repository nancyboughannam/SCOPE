# Licensing notice

This repository contains two things under two different licenses. Here is
what that means in plain terms.

## The two parts

**`FLaskAPIs/` and everything else at the top level** (the Python
optimization services, training scripts, ablation/statistics scripts, and
this documentation) is SCOPE's own original work. It is licensed under the
**MIT License** (see `LICENSE`). You can reuse, modify, and redistribute it
freely, including in closed-source or commercial projects, as long as you
keep the copyright notice.

**`ThreeBrains/`** is a modified copy of
[EdgeCloudSim](https://github.com/CagataySonmez/EdgeCloudSim), a Java
simulator originally created at Bogazici University. It is licensed under the
**GNU General Public License v3.0** (see `ThreeBrains/LICENSE`). Several
files in this directory — most importantly
`ThreeBrains/src/edu/boun/edgecloudsim/applications/sample_app5/VehicularEdgeOrchestrator.java`,
which contains SCOPE's edge-steering logic — are modifications of GPLv3
code, which makes them GPLv3 too, whether or not they'd otherwise carry
their own license. **GPLv3 requires that anyone who distributes this code,
or a modified version of it, also makes the source available under GPLv3.**

## Why two licenses instead of one

MIT and GPLv3 conflict if you try to combine them into a single distributed
program. We're able to keep them separate because SCOPE's Python services
and the Java simulator are not compiled or linked together — they are
independent processes that talk to each other over a local HTTP REST API
(see the README's architecture section). Under the FSF's own guidance, code
that merely communicates with GPL software over a process boundary like this
is not considered a "combined work," so it does not have to inherit the
GPL license.

**In practice, if you reuse this repository:**
- You can take the `FLaskAPIs/` Python code and use it under the permissive
  MIT terms — in a proprietary product, a different research project,
  wherever.
- If you take `ThreeBrains/` — including if you fork it, extend it, or
  redistribute a modified build of the simulator — you must comply with
  GPLv3, meaning your version has to stay open source under GPLv3 too.
- If you build a new combined tool that statically links or directly
  embeds `ThreeBrains/` Java code into another program (rather than
  calling it over HTTP the way SCOPE does), that new program likely also
  needs to be GPLv3-compatible. This is a legal judgment call, not
  something we can certify for you — talk to your institution's
  legal/IP office if this matters for your use case.

## What we are not claiming

This is our best-effort explanation of how the two licenses coexist in this
repository, based on the standard "mere aggregation over a network
interface" reading of GPLv3. It is not legal advice, and it has not been
reviewed by a lawyer. If licensing correctness matters to you (e.g., you
work at a company evaluating this code, or you plan to build a product on
top of it), get your own legal review before relying on this notice.
