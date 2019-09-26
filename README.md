# Kashen


## What's this?
- Experimental playground for specialization project
- The purpose is not to benchmark the application itself, but how the different caching solutions perform for a modern web application

### Todos
- [ ] Baseline benchmarking without any caching
  - [ ] Document oriented NoSQL
  - [ ] PostgreSQL
- [ ] Benchmark latency with cache with different databases
  - [ ] Document oriented NoSQL
  - [ ] PostgreSQL
- [ ] Benchmark with different policies
  - [ ] Redis
    - [ ] LRU
    - [ ] LFU
    - [ ] Random
    - [ ] Volatile LFU
    - [ ] Volatile TTL
    - [ ] Volatile LFU
    - [ ] Volatile random
  - [ ] Memcached
    - [ ] LRU
