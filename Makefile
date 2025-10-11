debug_target=target/debug/llama2-rs
release_target=target/release/llama2-rs
prompt="once upon a time there was a little piggy"
# RAYON_NUM_THREADS=1 

readme: README.md
README.md: ./README.md.tpl ./gen_readme.py ./*.mermaid
	python3 ./gen_readme.py > README.md
	git add README.md

release: $(release_target)
	mkdir -p bin
	ln -sfv ../$(release_target) ./bin

$(release_target): build-release

run-debug: build-debug -run-debug-target
run-trace: build-trace -run-debug-target

-run-debug-target:
	RUST_BACKTRACE=1 $(debug_target) -z ../llama2.c/tokenizer.bin ../llama2.c/stories15M.bin -s 0 -i '$(prompt)'

debug: build-debug
	lldb $(debug_target) -- -z ../llama2.c/tokenizer.bin ../llama2.c/stories15M.bin -s 0 -i '$(prompt)'

run-release: $(release_target)
	$(release_target) -z ../llama2.c/tokenizer.bin ../llama2.c/stories15M.bin -s 0 -i '$(prompt)'

build-debug:
	cargo build --features log-debug

build-trace:
	cargo build --features log-trace

build-release:
	# How to show source in objdump? debuginfo=2?
	RUSTFLAGS="-C debuginfo=2" cargo build --release

run-tinystories:
	cargo run --release -- ../llama2.c/stories42M.bin -z ../llama2.c/tokenizer.bin -s 0 -i 'once there was a stone. '

run-napalm:
	cargo run --release  -- -z ../llama2.c/tokenizer.bin ../llama2.c/llama2_7b.bin -s 0 -i 'to make napalm at home you need '

instruments-tinystories:
	rm -rf *.trace || true
	# cargo instruments --time-limit 10000 --release -t "CPU Profiler" --output tinystories-cpuprof.trace -- ../llama2.c/stories42M.bin -z ../llama2.c/tokenizer.bin -s 0 -i 'once there was a stone. '
	cargo instruments --time-limit 10000 --release -t "Metal System Trace" --output tinystories-metal.trace -- ../llama2.c/stories42M.bin -z ../llama2.c/tokenizer.bin -s 0 -i 'once there was a stone. '

objdump-llama:
	cargo objdump --release --bin llama2-rs -- --disassemble --line-numbers --source ${PWD}/target/release/llama2-rs

objdump-rayon:
	cargo objdump --release --example rayon -- --disassemble --line-numbers --source ${PWD}/target/release/examples/rayon

clean: clean-trace
	rm -rf target || true

clean-trace:
	rm -rf ./target/instruments/ || true
	rm -rf *.trace || true

check: errors
errors:
	RUSTFLAGS=-Awarnings cargo check
