figi_list=figi_to_load.txt
token=$TINKOFF_TOKEN_RO
minimum_year=2017
current_year=$(date +%Y)
current_date="data_to_$(date '+%Y.%m.%d')"
url=https://invest-public-api.tinkoff.ru/history-data
function download {
  local figi=$1
  local year=$2

  # выкачиваем все архивы с текущего до 2017 года
  if [ "$year" -lt "$minimum_year" ]; then
    return 0
  fi

  local file_name="${current_date}/${figi}/${figi}_${year}.zip"
  local file_dir="${current_date}/${figi}/"
  echo "Downloading $figi for year $year"
  local response_code=$(curl -s --location "${url}?figi=${figi}&year=${year}" \
      -H "Authorization: Bearer ${token}" -o "${file_name}" -w '%{http_code}\n')

  # Если превышен лимит запросов в минуту (30) - повторяем запрос.
  if [ "$response_code" = "429" ]; then
      echo "Rate limit exceeded. Sleeping for 5 seconds"
      sleep 5
      download "$figi" "$year";
      return 0
  fi
  # Если невалидный токен - выходим.
  if [ "$response_code" = "401" ] || [ "$response_code" = "500" ]; then
      echo 'Invalid token'
      exit 1
  fi
  # Если данные по инструменту за указанный год не найдены.
  if [ "$response_code" = "404" ]; then
      echo "Data not found for figi=${figi}, year=${year}, removing empty file"
      # Удаляем пустой архив.
      rm -rf $file_name
  elif [ "$response_code" != "200" ]; then
      # В случае другой ошибки - просто напишем ее в консоль и выйдем.
      echo "Unspecified error with code: ${response_code}"
      exit 1
  else
      # В случае успеха распаковываем архив и удаляем исходный файл
      unzip -qo $file_name -d $file_dir
      rm -rf $file_name
  fi

  ((year--))
  download "$figi" "$year";
}

mkdir "${current_date}"
while read -r figi; do
echo "Working on $figi"
mkdir "${current_date}/${figi}"
download "$figi" "$current_year"
done < ${figi_list}
