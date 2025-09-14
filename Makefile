debug_target=target/debug/llama2-rs
release_target=target/release/llama2-rs
prompt="once upon a time there was a little piggy"
# RAYON_NUM_THREADS=1 

release: $(release_target)
	mkdir -p bin
	ln -sfv ../$(release_target) ./bin

$(release_target): build-release

run: build
	RUST_BACKTRACE=1 $(debug_target) -z ../llama2.c/tokenizer.bin ../llama2.c/stories15M.bin -s 0 -i '$(prompt)'

debug: build
	lldb $(debug_target) -- -z ../llama2.c/tokenizer.bin ../llama2.c/stories15M.bin -s 0 -i '$(prompt)'

run-release: $(release_target)
	$(release_target) -z ../llama2.c/tokenizer.bin ../llama2.c/stories15M.bin -s 0 -i '$(prompt)'

build:
	cargo build --features debug

build-release:
	cargo build --release
