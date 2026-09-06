# Shared literal KEY=value reader; no expansion or execution of file values.
load_local_env() {
  local configuration_file entry key value
configuration_file="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -f $configuration_file ]]; then
  while IFS= read -r entry || [[ -n $entry ]]; do
    entry=${entry%$'\r'}
    entry="${entry#"${entry%%[![:space:]]*}"}"
    entry="${entry%"${entry##*[![:space:]]}"}"
    [[ -z $entry || $entry == \#* ]] && continue
    if [[ ! $entry =~ ^([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      echo "Invalid .env entry; expected KEY=value" >&2
      return 1
    fi
    key=${BASH_REMATCH[1]}
    value=${BASH_REMATCH[2]}
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ ${value:0:1} == \" || ${value:0:1} == \' ]]; then
      if [[ ${#value} -lt 2 || ${value: -1} != "${value:0:1}" ]]; then
        echo "Unmatched quote in .env" >&2
        return 1
      fi
      value=${value:1:${#value}-2}
    fi
    if [[ -n $value && ! -v $key ]]; then export "$key=$value"; fi
  done < "$configuration_file"
fi

}
load_local_env
